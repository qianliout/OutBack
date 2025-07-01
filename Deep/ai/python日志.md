以下是针对您需求的Python `logging` 库详细用法说明，包含重点功能实现：

***

## 一、基础功能实现

### 1. 输出到文件和控制台

    import logging
    import sys

    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 设置最低日志级别

        # 自定义格式（包含时间、名称、级别、消息）
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 控制台Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上
        console_handler.setFormatter(formatter)

        # 文件Handler
        file_handler = logging.FileHandler('app.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)    # 文件记录所有级别
        file_handler.setFormatter(formatter)

        # 避免重复添加Handler
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    # 使用示例
    logger = setup_logger(__name__)
    logger.info("系统启动")  # 同时输出到控制台和文件

### 2. 日志文件轮转

    from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

    # 按文件大小轮转（每个文件最大10MB，保留3个备份）
    rotating_handler = RotatingFileHandler(
        'app.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )

    # 按时间轮转（每天午夜轮转，保留7天）
    timed_handler = TimedRotatingFileHandler(
        'app.log',
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )

***

## 二、高级功能实现

### 3. 业务流程上下文透传（流水日志）

    import logging
    from contextvars import ContextVar
    import uuid

    # 创建上下文变量
    request_id = ContextVar('request_id')
    user_id = ContextVar('user_id')

    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.request_id = request_id.get(None)  # 获取当前上下文值
            record.user_id = user_id.get(None)
            return True

    # 配置Logger
    logger = logging.getLogger(__name__)
    logger.addFilter(ContextFilter())
    formatter = logging.Formatter(
        '%(asctime)s [%(request_id)s][user:%(user_id)s] %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 使用示例
    def process_request(user):
        # 设置上下文
        token = request_id.set(str(uuid.uuid4()))
        user_token = user_id.set(user.id)
        
        try:
            logger.info("开始处理请求")
            # ...业务逻辑...
            logger.debug("数据库查询完成")
        finally:
            # 清理上下文
            request_id.reset(token)
            user_id.reset(user_token)

    # 调用示例
    class User:
        def __init__(self, id):
            self.id = id

    process_request(User(12))  # 所有日志自动携带userID和requestID

### 4. 临时修改日志级别

    import logging
    import time
    from threading import Timer

    class LogLevelTemporarily:
        def __init__(self, logger, temp_level, duration):
            self.original_level = logger.getEffectiveLevel()
            self.logger = logger
            self.temp_level = temp_level
            self.duration = duration

        def __enter__(self):
            self.logger.setLevel(self.temp_level)
            # 自动恢复定时器
            self.timer = Timer(self.duration, self._restore)
            self.timer.start()

        def __exit__(self, *args):
            self.timer.cancel()
            self._restore()

        def _restore(self):
            self.logger.setLevel(self.original_level)

    # 使用示例
    logger = logging.getLogger(__name__)

    with LogLevelTemporarily(logger, logging.DEBUG, 60):
        logger.debug("这一分钟内可以显示DEBUG日志")  # ✅
        time.sleep(60)

    logger.debug("恢复后不再显示")  # ❌ 根据原始级别决定

### 5. 自定义日志格式

    import logging

    # 自定义格式包含：线程ID、模块行号、自定义字段
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname).1s] '
        '[%(threadName)s:%(module)s:%(lineno)d] '
        '{%(custom_key)s} - %(message)s'
    )

    # 添加自定义字段（通过过滤器）
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.custom_key = "CUSTOM_VALUE"  # 可替换为动态值
            return super().format(record)

    # 或通过LogRecord工厂
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.custom_key = "DYNAMIC_VALUE"  # 动态生成值
        return record

    logging.setLogRecordFactory(record_factory)

***

## 三、关键注意事项

### 1. 性能优化

*   ​**​避免高频debug日志​**​：生产环境慎用`DEBUG`级别
*   ​**​使用惰性格式化​**​：

        logger.info("User: %s, Action: %s", user_id, action)  # ✅ 正确
        logger.info(f"User: {user_id}")  # ❌ 立即求值

### 2. 多进程/多线程

*   ​**​进程安全​**​：使用`QueueHandler`和`QueueListener`
*   ​**​线程上下文​**​：使用`threading.local()`或`contextvars`

### 3. 配置管理

*   ​**​推荐使用配置字典​**​：

        import logging.config

        config = {
            'version': 1,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s %(levelname)s %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'app.log',
                    'formatter': 'detailed',
                    'maxBytes': 1024 * 1024
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['file']
            }
        }

        logging.config.dictConfig(config)

### 4. 异常处理

*   ​**​完整堆栈记录​**​：

        try:
            risky_operation()
        except Exception:
            logger.exception("操作失败")  # 自动记录异常堆栈

### 5. 日志文件管理

*   ​**​权限设置​**​：确保应用有日志文件写入权限
*   ​**​轮转策略​**​：根据业务需求选择大小轮转或时间轮转
*   ​**​日志清理​**​：结合crontab定期清理旧日志

***

## 四、完整示例整合

    import logging
    import sys
    from logging.handlers import RotatingFileHandler
    from contextvars import ContextVar
    import uuid

    # 上下文变量
    request_ctx = ContextVar('request_context')

    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.request_id = request_ctx.get(None) or '-'
            return True

    def configure_logging():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(request_id)s] '
            '[%(levelname).1s] %(message)s'
        )

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(ContextFilter())
        console_handler.setFormatter(formatter)

        # 文件轮转（每天一个文件，保留7天）
        file_handler = logging.handlers.TimedRotatingFileHandler(
            'app.log', when='midnight', backupCount=7, encoding='utf-8'
        )
        file_handler.addFilter(ContextFilter())
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def business_flow():
        token = request_ctx.set(str(uuid.uuid4()))
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("开始业务流程")
            logger.debug("详细调试信息")
            # ...业务逻辑...
        finally:
            request_ctx.reset(token)

    if __name__ == "__main__":
        configure_logging()
        business_flow()

***

## 五、总结

1.  ​**​多输出目标​**​：通过添加多个Handler实现
2.  ​**​日志轮转​**​：使用`RotatingFileHandler`或`TimedRotatingFileHandler`
3.  ​**​上下文透传​**​：结合`ContextVar`和自定义Filter
4.  ​**​动态日志级别​**​：通过上下文管理器实现临时修改
5.  ​**​格式定制​**​：继承`Formatter`类或修改LogRecord

以上方案满足企业级应用的日志需求，兼顾灵活性和性能。实际使用中可根据业务需求调整过滤策略和日志格式。
