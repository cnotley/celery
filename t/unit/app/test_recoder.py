import io
from unittest.mock import patch

import pytest
from celery import Celery, states
from celery.app.trace import build_tracer


def trace(app, task, args=(), kwargs=None, propagate=False,
          eager=True, request=None, task_id='id-rr', **opts):
    if kwargs is None:
        kwargs = {}
    tracer = build_tracer(task.name, task, eager=eager, propagate=propagate, app=app, **opts)
    ret = tracer(task_id, args, kwargs, request)
    return ret.retval, ret.info


class TestRecoder:
    def setup_method(self) -> None:
        self.app = Celery('recoder-app')

    def test_slow_task_logging(self):
        self.app.conf.update(time_threshold=0.0)
        err_buf = io.StringIO()
        self.app.task_logger.set_stderr(err_buf)

        @self.app.task(shared=False)
        def fast_task():
            return 'fast'

        trace(self.app, fast_task, ())
        assert 'slow' in err_buf.getvalue()

    def test_task_specific_threshold(self):
        self.app.conf.update(time_threshold=10.0)
        err_buf = io.StringIO()
        self.app.task_logger.set_stderr(err_buf)

        @self.app.task(shared=False)
        def just_task():
            return 'done'
        just_task.task_time_threshold = 0.0
        trace(self.app, just_task, ())
        assert 'slow' in err_buf.getvalue()

    def test_average_time_logging(self):
        self.app.conf.update(time_threshold=0.0, use_avg_time=True)
        err_buf = io.StringIO()
        self.app.task_logger.set_stderr(err_buf)

        @self.app.task(shared=False)
        def noop():
            return 'noop'
        trace(self.app, noop, ())
        trace(self.app, noop, ())
        assert 'avg-slow' in err_buf.getvalue()

    def test_auto_retry_for_slow(self):
        self.app.conf.update(time_threshold=0.0, task_always_eager=True)
        out_buf = io.StringIO()
        self.app.task_logger.set_stdout(out_buf)

        @self.app.task(bind=True, shared=False)
        def maybe_retry(self):
            return 'value'

        maybe_retry.use_auto_retry = True

        def fake_retry(*_a, **_k):
            maybe_retry.request.retries = getattr(maybe_retry.request, 'retries', 0) + 1
            raise states.RETRY

        with patch.object(maybe_retry, 'retry', side_effect=fake_retry) as mock_retry:
            with pytest.raises(states.RETRY):
                maybe_retry.delay()
            assert mock_retry.called

        assert 'Auto-retry' in out_buf.getvalue()