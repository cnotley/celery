import io
import pytest
import time
import re
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock
from celery import Celery, Task, TaskTracer, TaskTimer, RetryManager, TaskLogger
from multiprocessing import Manager
from celery.exceptions import Retry
from io import StringIO


def test_slow_task():
    app = Celery('timer_test')
    app.conf.update(task_always_eager=True, time_threshold=0.5)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(name="slow_task", bind=True)
    def slow_task(self):
        time.sleep(0.55)
        return 'done'

    result = slow_task.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    assert f"Task slow_task[{result.id}] is slow:" in log_stream.read()

def test_fast_task():
    app = Celery('timer_test_fast')
    app.conf.update(task_always_eager=True, time_threshold=0.5)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task
    def fast_task():
        return 'done'

    result = fast_task.delay()
    assert result.get(timeout=5) == 'done'

    log_stream.seek(0)
    assert "is slow" not in log_stream.read()

def test_default_threshold():
    app = Celery('default_threshold_test')
    app.conf.update(task_always_eager=True)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(bind=True)
    def slow_task_default(self):
        time.sleep(1.0)
        return 'done'

    result = slow_task_default.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    output = log_stream.read()
    assert "is slow" not in output


def test_no_stdout():
    app = Celery('no_stdout_test')
    app.conf.update(task_always_eager=True, time_threshold=0.5)

    stdout_stream = io.StringIO()
    app.task_logger.set_stdout(stdout_stream)

    @app.task(bind=True)
    def slow_task(self):
        time.sleep(0.6)
        return 'done'

    result = slow_task.delay()
    result.get()

    stdout_stream.seek(0)
    assert "is slow" not in stdout_stream.read()



def test_multiple_slow_tasks():
    app = Celery('multiple_tasks_test')
    app.conf.update(task_always_eager=True, time_threshold=0.3)

    fake_err = StringIO()
    app.task_logger.set_stderr(fake_err)

    @app.task(bind=True)
    def slow_task(self):
        time.sleep(0.4)
        return 'done'

    for _ in range(3):
        result = slow_task.delay()
        assert result.get() == 'done'

    output = fake_err.getvalue()
    occurrences = output.count("is slow")
    assert occurrences == 3, f"Expected 3 slow logs, got {occurrences}"


def test_aggressive_threshold():
    app = Celery('aggressive_threshold_test')
    app.conf.update(task_always_eager=True, time_threshold=0.001)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(bind=True)
    def any_task(self):
        time.sleep(0.002)
        return 'done'

    result = any_task.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    assert "is slow" in log_stream.read()

def test_aggressive_threshold_nolog():
    app = Celery('aggressive_threshold_test_nolog')
    app.conf.update(task_always_eager=True, time_threshold=0.005)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(bind=True)
    def fast_task(self):
        time.sleep(0.004)
        return 'done'

    result = fast_task.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    assert "is slow" not in log_stream.read(), "Unexpected slow log found"

def test_task_time_threshold():
    app = Celery('task_specific_threshold')
    app.conf.update(task_always_eager=True, time_threshold=0.5)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    class CustomTask(Task):
        task_time_threshold = 1.0

    @app.task(base=CustomTask, bind=True)
    def slow_task(self):
        time.sleep(0.9)
        return 'done'

    result = slow_task.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    assert "is slow" not in log_stream.read(), "Expected no slow log (task threshold should override app)"
    
def test_task_time_threshold_strict():
    app = Celery('stricter_task_threshold')
    app.conf.update(task_always_eager=True, time_threshold=1.0)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    class StrictTask(Task):
        task_time_threshold = 0.5

    @app.task(base=StrictTask, bind=True)
    def slow_task(self):
        time.sleep(0.6)
        return 'done'

    result = slow_task.delay()
    assert result.get() == 'done'

    log_stream.seek(0)
    assert "is slow" in log_stream.read(), "Expected slow log (task threshold should apply)"
    
def test_task_time_threshold_zero():
    app = Celery('zero_threshold_task')
    app.conf.update(task_always_eager=True, time_threshold=10.0)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    class ZeroThresholdTask(Task):
        task_time_threshold = 0.0  # Should override and mark all as slow

    @app.task(base=ZeroThresholdTask, bind=True)
    def fast_task(self):
        time.sleep(0.01)  # duration > 0.0
        return 'done'

    @app.task(base=ZeroThresholdTask, bind=True)
    def instant_task(self):
        return 'instant'

    fast_result = fast_task.delay()
    instant_result = instant_task.delay()

    assert fast_result.get() == 'done'
    assert instant_result.get() == 'instant'

    log_stream.seek(0)
    log_output = log_stream.read()

    assert log_output.count("is slow") >= 2, (
        f"Expected both tasks to be marked slow, log output:\n{log_output}"
    )

def test_avg():
    app = Celery("avg_system_triggered")
    app.conf.update(task_always_eager=True, use_avg_time=True, time_threshold=0.3)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(name="avg_test_task", bind=True)
    def t(self, delay):
        time.sleep(delay)
        return "ok"

    for _ in range(3):
        r = t.delay(0.1)
        assert r.get() == "ok"

    r = t.delay(1)
    assert r.get() == "ok"

    log_stream.seek(0)
    err = log_stream.read()

    assert "is avg-slow:" in err
    pattern = re.compile(
        rf"Task avg_test_task\[{r.id}\] is avg-slow: (\d+\.\d{{4}})s > threshold (\d+\.\d{{4}})s"
    )
    match = pattern.search(err)
    assert match, f"Unexpected log format: {err}"
    
def test_avg_fast():
    app = Celery("avg_system_safe")
    app.conf.update(task_always_eager=True, use_avg_time=True, time_threshold=0.35)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(name="avg_test_task_safe", bind=True)
    def t(self, delay):
        time.sleep(delay)
        return "ok"

    for _ in range(4):
        r = t.delay(0.3)
        assert r.get() == "ok"

    log_stream.seek(0)
    assert "is avg-slow:" not in log_stream.read()
    
def test_avg_with_custom_threshold():
    app = Celery("avg_system_task_level")
    app.conf.update(task_always_eager=True, use_avg_time=True, time_threshold=3)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    class CustomTask(Task):
        task_time_threshold = 0.2

    @app.task(name="avg_custom_threshold_task", base=CustomTask, bind=True)
    def t(self, delay):
        time.sleep(delay)
        return "ok"

    for _ in range(4):
        r = t.delay(0.25)
        assert r.get() == "ok"

    log_stream.seek(0)
    assert "is avg-slow:" in log_stream.read()

def test_auto_retry_on_timeout():
    app = Celery("auto_retry_test")
    app.conf.update(
        task_always_eager=True,
        use_auto_retry=True,
        time_threshold=0.3
    )

    stdout_stream = io.StringIO()
    app.task_logger.set_stdout(stdout_stream)

    @app.task(name="auto_retry_task", bind=True)
    def slow_task(self):
        time.sleep(0.4)
        return "done"

    with patch.object(slow_task, 'retry', side_effect=TimeoutError("forced")) as mock_retry:
        try:
            slow_task.delay()
        except TimeoutError:
            assert mock_retry.called, "Expected retry to be called"

    stdout_stream.seek(0)
    assert "Auto-retry: Task auto_retry_task[" in stdout_stream.read()


def test_auto_retry_safe():
    app = Celery("auto_retry_safe_task")
    app.conf.update(
        task_always_eager=True,
        use_auto_retry=True,
        time_threshold=1.0
    )

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(name="auto_safe_task", bind=True)
    def t(self):
        time.sleep(0.9)
        return "ok"

    with patch.object(t, 'retry') as mock_retry:
        result = t.delay()
        assert result.get() == "ok"
        assert not mock_retry.called, "Retry should not be called for fast task"

    log_stream.seek(0)
    assert "Auto-retry" not in log_stream.read()

    
def test_avg_custom_threshold_with_auto_retry():
    app = Celery("avg_custom_retry_test")
    app.conf.update(
        task_always_eager=True,
        use_avg_time=True,
        use_auto_retry=True,
        time_threshold=3.0
    )

    log_stream_out = io.StringIO()
    log_stream_err = io.StringIO()
    app.task_logger.set_stdout(log_stream_out)
    app.task_logger.set_stderr(log_stream_err)

    class CustomTask(Task):
        task_time_threshold = 0.2

    @app.task(name="avg_custom_retry_task", base=CustomTask, bind=True)
    def t(self, delay):
        time.sleep(delay)
        return "ok"

    for _ in range(3):
        r = t.delay(0.1)
        assert r.get() == "ok"

    with patch.object(t, 'retry', side_effect=Retry("forced")) as mock_retry:
        with pytest.raises(Retry):
            t.delay(0.6)
        assert mock_retry.called

    assert "is avg-slow:" in log_stream_err.getvalue()
    assert "Auto-retry: Task avg_custom_retry_task[" in log_stream_out.getvalue()
    
# SRP tests
# TaskTimer
def test_task_timer_is_slow_false():
    timer = TaskTimer(threshold=0.05, use_avg_time=False)
    timer.start()
    time.sleep(0.04)
    timer.stop()
    assert not timer.is_slow()

def test_task_timer_is_slow_true():
    timer = TaskTimer(threshold=0.05, use_avg_time=False)
    timer.start()
    time.sleep(0.06)
    timer.stop()
    assert timer.is_slow()
    
def test_task_timer_multiple_runs_average_time():
    timer = TaskTimer(threshold=0.05, use_avg_time=True)

    # Run 1 (fast)
    timer.start()
    time.sleep(0.01)
    timer.stop()
    assert timer.is_slow() is False

    # Run 2 (still under threshold, average ~0.02)
    timer.start()
    time.sleep(0.03)
    timer.stop()
    assert timer.is_slow() is False

    # Run 3 (slow enough to pull average over threshold)
    timer.start()
    time.sleep(0.2)
    timer.stop()
    assert timer.is_slow() is True

# RetryManager
def test_RetryManager_retry_if_enabled():
    task = MagicMock()
    task.request.retries = 1
    manager = RetryManager(use_auto_retry=True, max_retry_num=3)

    assert manager.should_retry(task) is True

def test_RetryManager_should_not_retry():
    task = MagicMock()
    task.request.retries = 1
    manager = RetryManager(use_auto_retry=False, max_retry_num=3)

    assert manager.should_retry(task) is False

def test_RetryManager_should_not_retry_if_exceeds_max():
    task = MagicMock()
    task.request.retries = 5
    manager = RetryManager(use_auto_retry=True, max_retry_num=3)

    assert manager.should_retry(task) is False

def test_RetryManager_retry_until_max_retry_num():
    task = MagicMock()
    task.retry = MagicMock()

    manager = RetryManager(use_auto_retry=True, max_retry_num=3)

    # Case 1: retries = 0 → should call task.retry()
    task.request.retries = 0
    manager.retry(task)
    task.retry.assert_called_once()

    # Case 2: retries = 3 (== max) → should NOT call task.retry()
    task.retry.reset_mock()
    task.request.retries = 3
    manager.retry(task)
    task.retry.assert_not_called()

    # Case 3: retries > max → should NOT call task.retry()
    task.retry.reset_mock()
    task.request.retries = 5
    manager.retry(task)
    task.retry.assert_not_called()

# TaskLogger
@pytest.fixture
def logger_streams():
    return io.StringIO(), io.StringIO()

def test_TaskLogger_log_failure(logger_streams):
    stdout, stderr = logger_streams
    logger = TaskLogger(stdout=stdout, stderr=stderr)
    logger.log_failure("task_name", "123", Exception("Some error"), "traceback_info")
    output = stderr.getvalue()
    assert "Task task_name[123] failed due to traceback_info" in output

def test_TaskLogger_log_slow(logger_streams):
    stdout, stderr = logger_streams
    logger = TaskLogger(stdout=stdout, stderr=stderr)
    logger.log_slow("task_name", "123", 3.5678, 2.0)
    output = stderr.getvalue()
    assert "Task task_name[123] is slow: 3.5678s > threshold 2.0000s" in output

def test_TaskLogger_log_avg_slow(logger_streams):
    stdout, stderr = logger_streams
    logger = TaskLogger(stdout=stdout, stderr=stderr)
    logger.log_slow("task_name", "123", 3.5678, 2.0, is_avg=True)
    output = stderr.getvalue()
    assert "Task task_name[123] is avg-slow: 3.5678s > threshold 2.0000s" in output

def test_TaskLogger_log_auto_retry(logger_streams):
    stdout, stderr = logger_streams
    logger = TaskLogger(stdout=stdout, stderr=stderr)

    logger.log_auto_retry("task_name", "123", 3.5678, 2.0, 1)
    logger.log_auto_retry("task_name", "123", 4.5678, 3.0, 2)

    output = stdout.getvalue()

    assert "Auto-retry: Task task_name[123] exceeded threshold 2.0000s with duration 3.5678s (retry 1)" in output
    assert "Auto-retry: Task task_name[123] exceeded threshold 3.0000s with duration 4.5678s (retry 2)" in output
    
# TaskTracer
def test_TaskTracer():
    timer = MagicMock(spec=TaskTimer)
    retry_manager = MagicMock(spec=RetryManager)
    logger = MagicMock(spec=TaskLogger)

    timer.start.return_value = None
    timer.stop.return_value = None
    timer.is_slow.return_value = True

    retry_manager.should_retry.return_value = True

    task_name = "mytask"
    task_id = "abc123"
    request = MagicMock()
    request.task = task_name
    request.id = task_id
    request.retries = 1

    tracer = TaskTracer(timer=timer, retry_manager=retry_manager, logger=logger)

    tracer.trace("uuid", args=[], kwargs={}, request=request)

    timer.start.assert_called_once()
    timer.stop.assert_called_once()
    timer.is_slow.assert_called_once_with()

    logger.log_slow.assert_called_once()
    logger.log_auto_retry.assert_called_once()

    retry_manager.should_retry.assert_called_once_with(request)
    retry_manager.retry.assert_called_once_with(request)

def test_concurrency():
    app = Celery("concurrency_test", broker="memory://", backend="rpc://")
    app.conf.update(task_always_eager=False, task_serializer="json")

    stdout, stderr = io.StringIO(), io.StringIO()
    execution_times = Manager().list()

    timer = TaskTimer(threshold=0.5, use_avg_time=False)
    retry_manager = RetryManager(use_auto_retry=True, max_retry_num=2)
    logger = TaskLogger(stdout=stdout, stderr=stderr)
    tracer = TaskTracer(timer=timer, retry_manager=retry_manager, logger=logger)

    @app.task(bind=True)
    def traced_task(self, duration):
        tracer.trace(
            uuid=self.request.id,
            args=(duration,),
            kwargs={},
            request=self.request,
        )
        time.sleep(duration)
        execution_times.append(duration)
        return duration

    durations = [0.4] * 5 + [0.6] * 5
    futures = [traced_task.apply_async(args=(d,)) for d in durations]
    results = [f.get(timeout=10) for f in futures]

    assert len(execution_times) == 10
    for result, expected in zip(results, durations):
        assert result == expected

    stdout_lines = stdout.getvalue().splitlines()
    stderr_lines = stderr.getvalue().splitlines()

    auto_retry_logs = [line for line in stdout_lines if "Auto-retry:" in line]
    slow_logs = [line for line in stderr_lines if "is slow:" in line]

    assert len(auto_retry_logs) == 5
    assert len(slow_logs) == 5

@pytest.mark.parametrize("max_retry_num", [1, 3, 5])
def test_max_retry_limit_enforced(max_retry_num):
    app = Celery("retry_limit_test")
    app.conf.update(task_always_eager=True)

    retry_manager = RetryManager(use_auto_retry=True, max_retry_num=max_retry_num)

    retry_counts = {"count": 0}

    class CustomRetryTask(Task):
        def run(self, *args, **kwargs):
            if retry_manager.should_retry(self):
                retry_counts["count"] += 1
                retry_manager.retry(self)
            return "done"

    @app.task(base=CustomRetryTask, bind=True)
    def retry_task(self):
        return self.run()

    with patch.object(CustomRetryTask, 'retry', side_effect=Retry("mock retry")) as mock_retry:
        with pytest.raises(Retry):
            retry_task()

    assert retry_counts["count"] == max_retry_num, (
        f"Expected {max_retry_num} retries, got {retry_counts['count']}"
    )
    
def test_time_threshold_runtime():
    app = Celery("runtime_threshold_test")
    app.conf.update(task_always_eager=True, time_threshold=1)

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)

    @app.task(bind=True)
    def measured_task(self, sleep_time):
        time.sleep(sleep_time)
        return "done"

    result1 = measured_task.delay(0.5)
    assert result1.get() == "done"

    log_stream.seek(0)
    output1 = log_stream.read()
    assert "is slow" not in output1, "Unexpected slow log before threshold change"

    log_stream = io.StringIO()
    app.task_logger.set_stderr(log_stream)
    app.conf.update(time_threshold=0.3)

    result2 = measured_task.delay(0.5)
    assert result2.get() == "done"

    log_stream.seek(0)
    output2 = log_stream.read()
    assert "is slow" in output2, "Expected slow log after threshold change"
