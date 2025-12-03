# tracer.py

import os
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


PROJECT_NAME = "tracing-research-agent"

# Register tracer provider
tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint="http://localhost:6006/v1/traces"
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)


