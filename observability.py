import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.gcp_trace.trace_exporter import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.gcp_monitoring import GoogleCloudMonitoringMetricsExporter
import time

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("sidekick")

# Tracing setup
def setup_tracing(service_name: str = "sidekick-adk"):
    """Initialize OpenTelemetry tracing and metrics"""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })
    
    # Trace provider
    trace_provider = TracerProvider(resource=resource)
    
    # GCP Cloud Trace exporter
    if os.getenv("ENABLE_GCP_TRACING", "false").lower() == "true":
        cloud_trace_exporter = CloudTraceSpanExporter()
        trace_provider.add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
        logger.info("GCP Cloud Trace enabled")
    
    trace.set_tracer_provider(trace_provider)
    
    # Metrics provider
    if os.getenv("ENABLE_GCP_METRICS", "false").lower() == "true":
        metrics_exporter = GoogleCloudMonitoringMetricsExporter()
        reader = PeriodicExportingMetricReader(metrics_exporter)
        metrics_provider = MeterProvider(resource=resource, metric_readers=[reader])
        set_meter_provider(metrics_provider)
        logger.info("GCP Cloud Monitoring enabled")
    
    return trace.get_tracer(__name__)

# Metrics
meter = get_meter_provider().get_meter("sidekick")
request_counter = meter.create_counter(
    "sidekick_requests_total",
    description="Total requests processed"
)
tool_usage_counter = meter.create_counter(
    "sidekick_tool_usage",
    description="Tool usage by type"
)
evaluation_score_histogram = meter.create_histogram(
    "sidekick_evaluation_scores",
    description="Distribution of evaluation scores"
)
session_duration = meter.create_histogram(
    "sidekick_session_duration_seconds",
    description="Session duration in seconds"
)

def traced_method(func_name: str):
    """Decorator for tracing methods"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(func_name) as span:
                span.set_attribute("function.name", func_name)
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration_seconds", duration)
        return wrapper
    return decorator

class ObservableAgent:
    """Mixin to add observability to agents"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)
        self.tracer = trace.get_tracer(__name__)
        
    def log_tool_usage(self, tool_name: str, success: bool, duration: float):
        """Log tool usage metrics"""
        tool_usage_counter.add(1, {
            "tool_name": tool_name,
            "agent": self.agent_name,
            "success": str(success)
        })
        self.logger.info(
            "tool_used",
            tool=tool_name,
            success=success,
            duration_seconds=duration
        )
