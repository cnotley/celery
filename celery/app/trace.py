"""Trace task execution.

This module defines how the task execution is traced:
errors are recorded, handlers are applied and so on.
"""
import logging
import os
import sys
import time
from collections import namedtuple
from warnings import warn

from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu.exceptions import EncodeError
from kombu.serialization import loads as loads_message
from kombu.serialization import prepare_accept_content
from kombu.utils.encoding import safe_repr, safe_str

from celery import current_app, group, signals, states
from typing import Optional
from celery._state import _task_stack
from celery.app.task import Context
from celery.app.task import Task as BaseTask
from celery.exceptions import BackendGetMetaError, Ignore, InvalidTaskError, Reject, Retry
from celery.result import AsyncResult
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import mro_lookup
from celery.utils.saferepr import saferepr
from celery.utils.serialization import get_pickleable_etype, get_pickleable_exception, get_pickled_exception

# ## ---
# This is the heart of the worker, the inner loop so to speak.
# It used to be split up into nice little classes and methods,
# but in the end it only resulted in bad performance and horrible tracebacks,
# so instead we now use one closure per task class.

# pylint: disable=redefined-outer-name
# We cache globals and attribute lookups, so disable this warning.
# pylint: disable=broad-except
# We know what we're doing...


__all__ = (
    'TraceInfo', 'build_tracer', 'trace_task',
    'setup_worker_optimizations', 'reset_worker_optimizations',
    'TaskTracer', 'TaskTimer', 'RetryManager', 'TaskLogger',
)

from celery.worker.state import successful_requests

logger = get_logger(__name__)

#: Format string used to log task receipt.
LOG_RECEIVED = """\
Task %(name)s[%(id)s] received\
"""

#: Format string used to log task success.
LOG_SUCCESS = """\
Task %(name)s[%(id)s] succeeded in %(runtime)ss: %(return_value)s\
"""

#: Format string used to log task failure.
LOG_FAILURE = """\
Task %(name)s[%(id)s] %(description)s: %(exc)s\
"""

#: Format string used to log task internal error.
LOG_INTERNAL_ERROR = """\
Task %(name)s[%(id)s] %(description)s: %(exc)s\
"""

#: Format string used to log task ignored.
LOG_IGNORED = """\
Task %(name)s[%(id)s] %(description)s\
"""

#: Format string used to log task rejected.
LOG_REJECTED = """\
Task %(name)s[%(id)s] %(exc)s\
"""

#: Format string used to log task retry.
LOG_RETRY = """\
Task %(name)s[%(id)s] retry: %(exc)s\
"""

log_policy_t = namedtuple(
    'log_policy_t',
    ('format', 'description', 'severity', 'traceback', 'mail'),
)

log_policy_reject = log_policy_t(LOG_REJECTED, 'rejected', logging.WARN, 1, 1)
log_policy_ignore = log_policy_t(LOG_IGNORED, 'ignored', logging.INFO, 0, 0)
log_policy_internal = log_policy_t(
    LOG_INTERNAL_ERROR, 'INTERNAL ERROR', logging.CRITICAL, 1, 1,
)
log_policy_expected = log_policy_t(
    LOG_FAILURE, 'raised expected', logging.INFO, 0, 0,
)
log_policy_unexpected = log_policy_t(
    LOG_FAILURE, 'raised unexpected', logging.ERROR, 1, 1,
)

send_prerun = signals.task_prerun.send
send_postrun = signals.task_postrun.send
send_success = signals.task_success.send
STARTED = states.STARTED
SUCCESS = states.SUCCESS
IGNORED = states.IGNORED
REJECTED = states.REJECTED
RETRY = states.RETRY
FAILURE = states.FAILURE
EXCEPTION_STATES = states.EXCEPTION_STATES
IGNORE_STATES = frozenset({IGNORED, RETRY, REJECTED})

#: set by :func:`setup_worker_optimizations`
_localized = []
_patched = {}

trace_ok_t = namedtuple('trace_ok_t', ('retval', 'info', 'runtime', 'retstr'))


# New classes for the recoder and object-oriented trace system.

class TaskTimer:
    """Utility for measuring task execution run time."""
    # Maintains cumulative durations for tasks when computing averages
    _durations = {}

    def __init__(self, threshold: float, use_avg_time: bool = False) -> None:
        self.threshold = threshold
        self.use_avg_time = use_avg_time
        self._start_time: Optional[float] = None
        self._last_duration: float = 0.0
        self.task_name: Optional[str] = None

    def start(self) -> None:
        """Mark start time of the task."""
        self._start_time = time.monotonic()

    def stop(self) -> float:
        """Stop timer and return elapsed duration."""
        if self._start_time is None:
            return 0.0
        end = time.monotonic()
        self._last_duration = end - self._start_time
        if self.use_avg_time and self.task_name:
            # Update cumulative stats for average computation
            total, count = TaskTimer._durations.get(self.task_name, (0.0, 0))
            total += self._last_duration
            count += 1
            TaskTimer._durations[self.task_name] = (total, count)
        return self._last_duration

    def is_slow(self) -> bool:
        """Determine if the last duration exceeds threshold (or average if configured)."""
        if self.use_avg_time and self.task_name:
            total, count = TaskTimer._durations.get(self.task_name, (0.0, 0))
            avg = total / count if count else 0.0
            return avg > self.threshold
        return self._last_duration > self.threshold

    @property
    def duration(self) -> float:
        return self._last_duration

    @property
    def average(self) -> float:
        if not self.use_avg_time or not self.task_name:
            return self._last_duration
        total, count = TaskTimer._durations.get(self.task_name, (0.0, 0))
        return total / count if count else 0.0


class RetryManager:
    """Controls automatic retry semantics for slow tasks."""
    def __init__(self, use_auto_retry: bool, max_retry_num: int) -> None:
        self.use_auto_retry = use_auto_retry
        self.max_retry_num = max_retry_num

    def should_retry(self, task: BaseTask) -> bool:
        """Return True if auto-retry enabled and retries so far < max."""
        try:
            retries = task.request.retries
        except AttributeError:
            retries = 0
        return self.use_auto_retry and retries < self.max_retry_num

    def retry(self, task: BaseTask) -> None:
        """Raise a retry for the given task."""
        task.retry()


class TaskLogger:
    """Simple logger interface for tasks to stdout/stderr."""
    def __init__(self, stdout=sys.stdout, stderr=sys.stderr) -> None:
        self._stdout = stdout
        self._stderr = stderr

    def set_stdout(self, stream) -> None:
        self._stdout = stream

    def set_stderr(self, stream) -> None:
        self._stderr = stream

    def log_failure(self, task_name: str, task_id: str, exc: Exception, traceback: str) -> None:
        self._stderr.write(f'Task {task_name}[{task_id}] failed due to {traceback}\n')
        self._stderr.flush()

    def log_slow(self, task_name: str, task_id: str, duration: float, threshold: float, is_avg: bool = False) -> None:
        label = 'avg-slow' if is_avg else 'slow'
        self._stderr.write(f'Task {task_name}[{task_id}] is {label}: {duration:.4f}s > threshold {threshold:.4f}s\n')
        self._stderr.flush()

    def log_auto_retry(self, task_name: str, task_id: str, duration: float, threshold: float, retry_count: int) -> None:
        self._stdout.write(f'Auto-retry: Task {task_name}[{task_id}] exceeded threshold {threshold:.4f}s '
                           f'with duration {duration:.4f}s (retry {retry_count})\n')
        self._stdout.flush()


class TaskTracer:
    """Core trace logic moved into object-oriented structure."""
    def __init__(self, timer: TaskTimer, retry_manager: RetryManager, logger: TaskLogger) -> None:
        self.timer = timer
        self.retry_manager = retry_manager
        self.logger = logger
        # attributes below will be populated by build_tracer
        self.fun = None
        self.task: Optional[BaseTask] = None
        self.name: Optional[str] = None
        self.loader_task_init = None
        self.loader_cleanup = None
        self.ignore_result = False
        self.track_started = False
        self.publish_result = False
        self.deduplicate_successful_tasks = False
        self.hostname = None
        self.inherit_parent_priority = False
        self.app = None
        self.monotonic = time.monotonic
        self.Info = TraceInfo
        self.trace_ok_t = trace_ok_t
        self.propagate = False
        self.eager = False
        self.prerun_receivers = []
        self.postrun_receivers = []
        self.success_receivers = []
        self.task_before_start = None
        self.task_on_success = None
        self.task_after_return = None
        self.request_stack = None
        self.push_request = None
        self.pop_request = None
        self.push_task = None
        self.pop_task = None
        self.does_info = False
        self.resultrepr_maxsize = None
        self.signature = None
        self.loader = None

    def on_error(self, request: Context, exc: Exception, state=FAILURE, call_errbacks: bool = True):
        """Handle error based on the given exception."""
        if self.propagate:
            raise
        I = self.Info(state, exc)
        R = I.handle_error_state(
            self.task, request, eager=self.eager, call_errbacks=call_errbacks,
        )
        return I, R, I.state, I.retval

    def trace(self, uuid: str, args, kwargs, request=None):
        """Replicates original trace_task closure."""
        R = I = T = Rstr = retval = state = None
        task_request = None
        time_start = self.monotonic()
        self.timer.start()
        try:
            try:
                kwargs.items
            except AttributeError:
                raise InvalidTaskError('Task keyword arguments is not a mapping')
            task_request = Context(request or {}, args=args, called_directly=False, kwargs=kwargs)
            redelivered = (task_request.delivery_info and task_request.delivery_info.get('redelivered', False))
            if self.deduplicate_successful_tasks and redelivered:
                if task_request.id in successful_requests:
                    return self.trace_ok_t(R, I, T, Rstr)
                r = AsyncResult(task_request.id, app=self.app)
                try:
                    state = r.state
                except BackendGetMetaError:
                    pass
                else:
                    if state == SUCCESS:
                        info(LOG_IGNORED, {
                            'id': task_request.id,
                            'name': get_task_name(task_request, self.name),
                            'description': 'Task already completed successfully.'
                        })
                        return self.trace_ok_t(R, I, T, Rstr)
            self.push_task(self.task)
            root_id = task_request.root_id or uuid
            task_priority = task_request.delivery_info.get('priority') if self.inherit_parent_priority else None
            self.push_request(task_request)
            try:
                # Pre
                if self.prerun_receivers:
                    send_prerun(sender=self.task, task_id=uuid, task=self.task,
                                args=args, kwargs=kwargs)
                self.loader_task_init(uuid, self.task)
                if self.track_started:
                    self.task.backend.store_result(
                        uuid, {'pid': os.getpid(), 'hostname': self.hostname}, STARTED,
                        request=task_request,
                    )
                try:
                    if self.task_before_start:
                        self.task_before_start(uuid, args, kwargs)
                    R = retval = self.fun(*args, **kwargs)
                    state = SUCCESS
                    # Stop timer after successful execution and calculate duration
                    self.timer.stop()
                    if self.timer.is_slow():
                        # compute duration for logging: average if configured, else last duration
                        duration_to_report = self.timer.average if self.timer.use_avg_time else self.timer.duration
                        self.logger.log_slow(get_task_name(task_request, self.name), uuid,
                                             duration_to_report,
                                             self.timer.threshold, is_avg=self.timer.use_avg_time)
                        if self.retry_manager.should_retry(self.task):
                            # Log and raise retry to integrate with retry handling below
                            current_retry = getattr(self.task.request, 'retries', 0) + 1
                            self.logger.log_auto_retry(get_task_name(task_request, self.name), uuid,
                                                       self.timer.duration, self.timer.threshold, current_retry)
                            self.retry_manager.retry(self.task)
                except Reject as exc:
                    I, R = self.Info(REJECTED, exc), ExceptionInfo(internal=True)
                    state, retval = I.state, I.retval
                    I.handle_reject(self.task, task_request)
                    traceback_clear(exc)
                except Ignore as exc:
                    I, R = self.Info(IGNORED, exc), ExceptionInfo(internal=True)
                    state, retval = I.state, I.retval
                    I.handle_ignore(self.task, task_request)
                    traceback_clear(exc)
                except Retry as exc:
                    I, R, state, retval = self.on_error(task_request, exc, RETRY, call_errbacks=False)
                    traceback_clear(exc)
                except Exception as exc:
                    I, R, state, retval = self.on_error(task_request, exc)
                    traceback_clear(exc)
                except BaseException:
                    raise
                else:
                    try:
                        # Apply callbacks and chains before storing result
                        callbacks = self.task.request.callbacks
                        if callbacks:
                            if len(callbacks) > 1:
                                sigs, groups = [], []
                                for sig in callbacks:
                                    sig = self.signature(sig, app=self.app)
                                    if isinstance(sig, group):
                                        groups.append(sig)
                                    else:
                                        sigs.append(sig)
                                for group_ in groups:
                                    group_.apply_async(
                                        (retval,),
                                        parent_id=uuid, root_id=root_id,
                                        priority=task_priority
                                    )
                                if sigs:
                                    group(sigs, app=self.app).apply_async(
                                        (retval,),
                                        parent_id=uuid, root_id=root_id,
                                        priority=task_priority
                                    )
                            else:
                                self.signature(callbacks[0], app=self.app).apply_async(
                                    (retval,), parent_id=uuid, root_id=root_id,
                                    priority=task_priority
                                )
                        chain = task_request.chain
                        if chain:
                            _chsig = self.signature(chain.pop(), app=self.app)
                            _chsig.apply_async(
                                (retval,), chain=chain,
                                parent_id=uuid, root_id=root_id,
                                priority=task_priority
                            )
                        self.task.backend.mark_as_done(
                            uuid, retval, task_request, self.publish_result,
                        )
                    except EncodeError as exc:
                        I, R, state, retval = self.on_error(task_request, exc)
                    else:
                        Rstr = saferepr(R, self.resultrepr_maxsize)
                        T = self.monotonic() - time_start
                        if self.task_on_success:
                            self.task_on_success(retval, uuid, args, kwargs)
                        if self.success_receivers:
                            send_success(sender=self.task, result=retval)
                        if self.does_info:
                            info(LOG_SUCCESS, {
                                'id': uuid,
                                'name': get_task_name(task_request, self.name),
                                'return_value': Rstr,
                                'runtime': T,
                                'args': task_request.get('argsrepr') or safe_repr(args),
                                'kwargs': task_request.get('kwargsrepr') or safe_repr(kwargs),
                            })
                # post
                if state not in IGNORE_STATES:
                    if self.task_after_return:
                        self.task_after_return(state, retval, uuid, args, kwargs, None)
            finally:
                try:
                    if self.postrun_receivers:
                        send_postrun(sender=self.task, task_id=uuid, task=self.task,
                                     args=args, kwargs=kwargs,
                                     retval=retval, state=state)
                finally:
                    self.pop_task()
                    self.pop_request()
                    if not self.eager:
                        try:
                            self.task.backend.process_cleanup()
                            self.loader_cleanup()
                        except (KeyboardInterrupt, SystemExit, MemoryError):
                            raise
                        except Exception as exc:
                            logger.error('Process cleanup failed: %r', exc,
                                         exc_info=True)
        except MemoryError:
            raise
        except Exception as exc:
            _signal_internal_error(self.task, uuid, args, kwargs, request, exc)
            if self.eager:
                raise
            R = report_internal_error(self.task, exc)
            if task_request is not None:
                I, _, _, _ = self.on_error(task_request, exc)
        return self.trace_ok_t(R, I, T, Rstr)


def info(fmt, context):
    """Log 'fmt % context' with severity 'INFO'.

    'context' is also passed in extra with key 'data' for custom handlers.
    """
    logger.info(fmt, context, extra={'data': context})


def task_has_custom(task, attr):
    """Return true if the task overrides ``attr``."""
    return mro_lookup(task.__class__, attr, stop={BaseTask, object},
                      monkey_patched=['celery.app.task'])


def get_log_policy(task, einfo, exc):
    if isinstance(exc, Reject):
        return log_policy_reject
    elif isinstance(exc, Ignore):
        return log_policy_ignore
    elif einfo.internal:
        return log_policy_internal
    else:
        if task.throws and isinstance(exc, task.throws):
            return log_policy_expected
        return log_policy_unexpected


def get_task_name(request, default):
    """Use 'shadow' in request for the task name if applicable."""
    # request.shadow could be None or an empty string.
    # If so, we should use default.
    return getattr(request, 'shadow', None) or default


class TraceInfo:
    """Information about task execution."""

    __slots__ = ('state', 'retval')

    def __init__(self, state, retval=None):
        self.state = state
        self.retval = retval

    def handle_error_state(self, task, req,
                           eager=False, call_errbacks=True):
        if task.ignore_result:
            store_errors = task.store_errors_even_if_ignored
        elif eager and task.store_eager_result:
            store_errors = True
        else:
            store_errors = not eager

        return {
            RETRY: self.handle_retry,
            FAILURE: self.handle_failure,
        }[self.state](task, req,
                      store_errors=store_errors,
                      call_errbacks=call_errbacks)

    def handle_reject(self, task, req, **kwargs):
        self._log_error(task, req, ExceptionInfo())

    def handle_ignore(self, task, req, **kwargs):
        self._log_error(task, req, ExceptionInfo())

    def handle_retry(self, task, req, store_errors=True, **kwargs):
        """Handle retry exception."""
        # the exception raised is the Retry semi-predicate,
        # and it's exc' attribute is the original exception raised (if any).
        type_, _, tb = sys.exc_info()
        try:
            reason = self.retval
            einfo = ExceptionInfo((type_, reason, tb))
            if store_errors:
                task.backend.mark_as_retry(
                    req.id, reason.exc, einfo.traceback, request=req,
                )
            task.on_retry(reason.exc, req.id, req.args, req.kwargs, einfo)
            signals.task_retry.send(sender=task, request=req,
                                    reason=reason, einfo=einfo)
            info(LOG_RETRY, {
                'id': req.id,
                'name': get_task_name(req, task.name),
                'exc': str(reason),
            })
            return einfo
        finally:
            del tb

    def handle_failure(self, task, req, store_errors=True, call_errbacks=True):
        """Handle exception."""
        orig_exc = self.retval

        exc = get_pickleable_exception(orig_exc)
        if exc.__traceback__ is None:
            # `get_pickleable_exception` may have created a new exception without
            # a traceback.
            _, _, exc.__traceback__ = sys.exc_info()

        exc_type = get_pickleable_etype(type(orig_exc))

        # make sure we only send pickleable exceptions back to parent.
        einfo = ExceptionInfo(exc_info=(exc_type, exc, exc.__traceback__))

        task.backend.mark_as_failure(
            req.id, exc, einfo.traceback,
            request=req, store_result=store_errors,
            call_errbacks=call_errbacks,
        )

        task.on_failure(exc, req.id, req.args, req.kwargs, einfo)
        signals.task_failure.send(sender=task, task_id=req.id,
                                  exception=exc, args=req.args,
                                  kwargs=req.kwargs,
                                  traceback=exc.__traceback__,
                                  einfo=einfo)
        self._log_error(task, req, einfo)
        return einfo

    def _log_error(self, task, req, einfo):
        eobj = einfo.exception = get_pickled_exception(einfo.exception)
        if isinstance(eobj, ExceptionWithTraceback):
            eobj = einfo.exception = eobj.exc
        exception, traceback, exc_info, sargs, skwargs = (
            safe_repr(eobj),
            safe_str(einfo.traceback),
            einfo.exc_info,
            req.get('argsrepr') or safe_repr(req.args),
            req.get('kwargsrepr') or safe_repr(req.kwargs),
        )
        policy = get_log_policy(task, einfo, eobj)
        context = {
            'hostname': req.hostname,
            'id': req.id,
            'name': get_task_name(req, task.name),
            'exc': exception,
            'traceback': traceback,
            'args': sargs,
            'kwargs': skwargs,
            'description': policy.description,
            'internal': einfo.internal,
        }
        # Use the new TaskLogger to emit failure logs
        try:
            current_app.task_logger.log_failure(context['name'], context['id'], exception, traceback)
        except Exception:
            # fall back to existing logger on any issues
            logger.log(policy.severity, policy.format.strip(), context,
                       exc_info=exc_info if policy.traceback else None,
                       extra={'data': context})


def traceback_clear(exc=None):
    # Cleared Tb, but einfo still has a reference to Traceback.
    # exc cleans up the Traceback at the last moment that can be revealed.
    tb = None
    if exc is not None:
        if hasattr(exc, '__traceback__'):
            tb = exc.__traceback__
        else:
            _, _, tb = sys.exc_info()
    else:
        _, _, tb = sys.exc_info()

    while tb is not None:
        try:
            tb.tb_frame.clear()
            tb.tb_frame.f_locals
        except RuntimeError:
            # Ignore the exception raised if the frame is still executing.
            pass
        tb = tb.tb_next


def build_tracer(name, task, loader=None, hostname=None, store_errors=True,
                 Info=TraceInfo, eager=False, propagate=False, app=None,
                 monotonic=time.monotonic, trace_ok_t=trace_ok_t,
                 IGNORE_STATES=IGNORE_STATES):
    """Return a callable tracing function for this task.

    Catches all exceptions and updates result backend with the
    state and result.

    If the call was successful, it saves the result to the task result
    backend, and sets the task status to `"SUCCESS"`.

    If the call raises :exc:`~@Retry`, it extracts
    the original exception, uses that as the result and sets the task state
    to `"RETRY"`.

    If the call results in an exception, it saves the exception as the task
    result, and sets the task state to `"FAILURE"`.

    Return a function that takes the following arguments:

        :param uuid: The id of the task.
        :param args: List of positional args to pass on to the function.
        :param kwargs: Keyword arguments mapping to pass on to the function.
        :keyword request: Request dict.

    """

    # pylint: disable=too-many-statements

    # If the task doesn't define a custom __call__ method
    # we optimize it away by simply calling the run method directly,
    # saving the extra method call and a line less in the stack trace.
    fun = task if task_has_custom(task, '__call__') else task.run

    loader = loader or app.loader
    ignore_result = task.ignore_result
    track_started = task.track_started
    track_started = not eager and (task.track_started and not ignore_result)

    # #6476
    if eager and not ignore_result and task.store_eager_result:
        publish_result = True
    else:
        publish_result = not eager and not ignore_result

    deduplicate_successful_tasks = ((app.conf.task_acks_late or task.acks_late)
                                    and app.conf.worker_deduplicate_successful_tasks
                                    and app.backend.persistent)

    hostname = hostname or gethostname()
    inherit_parent_priority = app.conf.task_inherit_parent_priority

    loader_task_init = loader.on_task_init
    loader_cleanup = loader.on_process_cleanup

    task_before_start = None
    task_on_success = None
    task_after_return = None
    if task_has_custom(task, 'before_start'):
        task_before_start = task.before_start
    if task_has_custom(task, 'on_success'):
        task_on_success = task.on_success
    if task_has_custom(task, 'after_return'):
        task_after_return = task.after_return

    pid = os.getpid()

    request_stack = task.request_stack
    push_request = request_stack.push
    pop_request = request_stack.pop
    push_task = _task_stack.push
    pop_task = _task_stack.pop
    _does_info = logger.isEnabledFor(logging.INFO)
    resultrepr_maxsize = task.resultrepr_maxsize

    prerun_receivers = signals.task_prerun.receivers
    postrun_receivers = signals.task_postrun.receivers
    success_receivers = signals.task_success.receivers

    from celery import canvas
    signature = canvas.maybe_signature  # maybe_ does not clone if already

    def on_error(request, exc, state=FAILURE, call_errbacks=True):
        if propagate:
            raise
        I = Info(state, exc)
        R = I.handle_error_state(
            task, request, eager=eager, call_errbacks=call_errbacks,
        )
        return I, R, I.state, I.retval

    # Determine timing and retry behaviours
    threshold = getattr(task, 'task_time_threshold') if hasattr(task, 'task_time_threshold') else app.conf.time_threshold
    use_avg_time = app.conf.use_avg_time
    timer = TaskTimer(threshold=threshold, use_avg_time=use_avg_time)
    timer.task_name = name
    # Determine auto-retry behaviour
    use_auto_retry = getattr(task, 'use_auto_retry') if hasattr(task, 'use_auto_retry') else app.conf.use_auto_retry
    max_retry_num = getattr(task, 'max_retry_num') if hasattr(task, 'max_retry_num') else app.conf.max_retry_num
    retry_manager = RetryManager(use_auto_retry=use_auto_retry, max_retry_num=max_retry_num)
    tracer_logger = app.task_logger if hasattr(app, 'task_logger') else TaskLogger()
    tracer = TaskTracer(timer=timer, retry_manager=retry_manager, logger=tracer_logger)
    # copy original closure state into the tracer
    tracer.fun = task if task_has_custom(task, '__call__') else task.run
    tracer.task = task
    tracer.name = name
    tracer.loader_task_init = loader.on_task_init if loader else app.loader.on_task_init
    tracer.loader_cleanup = loader.on_process_cleanup if loader else app.loader.on_process_cleanup
    tracer.ignore_result = task.ignore_result
    tracer.track_started = not eager and (task.track_started and not task.ignore_result)
    tracer.publish_result = not eager and not task.ignore_result if not eager else not task.ignore_result and task.store_eager_result
    tracer.deduplicate_successful_tasks = ((app.conf.task_acks_late or task.acks_late)
                                           and app.conf.worker_deduplicate_successful_tasks
                                           and app.backend.persistent)
    tracer.hostname = hostname or gethostname()
    tracer.inherit_parent_priority = app.conf.task_inherit_parent_priority
    tracer.monotonic = monotonic
    tracer.Info = Info
    tracer.trace_ok_t = trace_ok_t
    tracer.propagate = propagate
    tracer.eager = eager
    tracer.prerun_receivers = signals.task_prerun.receivers
    tracer.postrun_receivers = signals.task_postrun.receivers
    tracer.success_receivers = signals.task_success.receivers
    tracer.request_stack = task.request_stack
    tracer.push_request = task.request_stack.push
    tracer.pop_request = task.request_stack.pop
    tracer.push_task = _task_stack.push
    tracer.pop_task = _task_stack.pop
    tracer.does_info = logger.isEnabledFor(logging.INFO)
    tracer.resultrepr_maxsize = task.resultrepr_maxsize
    # maybe_signature does not clone if already a signature.
    from celery import canvas
    tracer.signature = canvas.maybe_signature
    if task_has_custom(task, 'before_start'):
        tracer.task_before_start = task.before_start
    if task_has_custom(task, 'on_success'):
        tracer.task_on_success = task.on_success
    if task_has_custom(task, 'after_return'):
        tracer.task_after_return = task.after_return
    tracer.app = app
    tracer.loader = loader or app.loader
    return tracer.trace


def trace_task(task, uuid, args, kwargs, request=None, **opts):
    """Trace task execution."""
    request = {} if not request else request
    try:
        if task.__trace__ is None:
            task.__trace__ = build_tracer(task.name, task, **opts)
        return task.__trace__(uuid, args, kwargs, request)
    except Exception as exc:
        _signal_internal_error(task, uuid, args, kwargs, request, exc)
        return trace_ok_t(report_internal_error(task, exc), TraceInfo(FAILURE, exc), 0.0, None)


def _signal_internal_error(task, uuid, args, kwargs, request, exc):
    """Send a special `internal_error` signal to the app for outside body errors."""
    try:
        _, _, tb = sys.exc_info()
        einfo = ExceptionInfo()
        einfo.exception = get_pickleable_exception(einfo.exception)
        einfo.type = get_pickleable_etype(einfo.type)
        signals.task_internal_error.send(
            sender=task,
            task_id=uuid,
            args=args,
            kwargs=kwargs,
            request=request,
            exception=exc,
            traceback=tb,
            einfo=einfo,
        )
    finally:
        del tb


def trace_task_ret(name, uuid, request, body, content_type,
                   content_encoding, loads=loads_message, app=None,
                   **extra_request):
    app = app or current_app._get_current_object()
    embed = None
    if content_type:
        accept = prepare_accept_content(app.conf.accept_content)
        args, kwargs, embed = loads(
            body, content_type, content_encoding, accept=accept,
        )
    else:
        args, kwargs, embed = body
    hostname = gethostname()
    request.update({
        'args': args, 'kwargs': kwargs,
        'hostname': hostname, 'is_eager': False,
    }, **embed or {})
    R, I, T, Rstr = trace_task(app.tasks[name],
                               uuid, args, kwargs, request, app=app)
    return (1, R, T) if I else (0, Rstr, T)


def fast_trace_task(task, uuid, request, body, content_type,
                    content_encoding, loads=loads_message, _loc=None,
                    hostname=None, **_):
    _loc = _localized if not _loc else _loc
    embed = None
    tasks, accept, hostname = _loc
    if content_type:
        args, kwargs, embed = loads(
            body, content_type, content_encoding, accept=accept,
        )
    else:
        args, kwargs, embed = body
    request.update({
        'args': args, 'kwargs': kwargs,
        'hostname': hostname, 'is_eager': False,
    }, **embed or {})
    R, I, T, Rstr = tasks[task].__trace__(
        uuid, args, kwargs, request,
    )
    return (1, R, T) if I else (0, Rstr, T)


def report_internal_error(task, exc):
    _type, _value, _tb = sys.exc_info()
    try:
        _value = task.backend.prepare_exception(exc, 'pickle')
        exc_info = ExceptionInfo((_type, _value, _tb), internal=True)
        warn(RuntimeWarning(
            'Exception raised outside body: {!r}:\n{}'.format(
                exc, exc_info.traceback)))
        return exc_info
    finally:
        del _tb


def setup_worker_optimizations(app, hostname=None):
    """Setup worker related optimizations."""
    hostname = hostname or gethostname()

    # make sure custom Task.__call__ methods that calls super
    # won't mess up the request/task stack.
    _install_stack_protection()

    # all new threads start without a current app, so if an app is not
    # passed on to the thread it will fall back to the "default app",
    # which then could be the wrong app.  So for the worker
    # we set this to always return our app.  This is a hack,
    # and means that only a single app can be used for workers
    # running in the same process.
    app.set_current()
    app.set_default()

    # evaluate all task classes by finalizing the app.
    app.finalize()

    # set fast shortcut to task registry
    _localized[:] = [
        app._tasks,
        prepare_accept_content(app.conf.accept_content),
        hostname,
    ]

    app.use_fast_trace_task = True


def reset_worker_optimizations(app=current_app):
    """Reset previously configured optimizations."""
    try:
        delattr(BaseTask, '_stackprotected')
    except AttributeError:
        pass
    try:
        BaseTask.__call__ = _patched.pop('BaseTask.__call__')
    except KeyError:
        pass
    app.use_fast_trace_task = False


def _install_stack_protection():
    # Patches BaseTask.__call__ in the worker to handle the edge case
    # where people override it and also call super.
    #
    # - The worker optimizes away BaseTask.__call__ and instead
    #   calls task.run directly.
    # - so with the addition of current_task and the request stack
    #   BaseTask.__call__ now pushes to those stacks so that
    #   they work when tasks are called directly.
    #
    # The worker only optimizes away __call__ in the case
    # where it hasn't been overridden, so the request/task stack
    # will blow if a custom task class defines __call__ and also
    # calls super().
    if not getattr(BaseTask, '_stackprotected', False):
        _patched['BaseTask.__call__'] = orig = BaseTask.__call__

        def __protected_call__(self, *args, **kwargs):
            stack = self.request_stack
            req = stack.top
            if req and not req._protected and \
                    len(stack) == 1 and not req.called_directly:
                req._protected = 1
                return self.run(*args, **kwargs)
            return orig(self, *args, **kwargs)
        BaseTask.__call__ = __protected_call__
        BaseTask._stackprotected = True
