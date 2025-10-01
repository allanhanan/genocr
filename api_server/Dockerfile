# Multi-stage build for GenOCR API Server
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies including jsoncpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopencv-dev \
    libssl-dev \
    uuid-dev \
    zlib1g-dev \
    libbrotli-dev \
    libjsoncpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy project files
COPY cpp_core/ ./cpp_core/
COPY api_server/ ./api_server/
COPY models/ ./models/

# Build cpp_core first
WORKDIR /build/cpp_core/build
RUN cmake .. && make -j$(nproc)

# Build api_server
WORKDIR /build/api_server/build
RUN cmake .. && make -j$(nproc)

# Runtime stage - minimal image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies with ALL OpenCV modules needed
RUN apt-get update && apt-get install -y \
    libopencv-core4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-imgproc4.5d \
    libopencv-dnn4.5d \
    libopencv-highgui4.5d \
    libssl3 \
    uuid-runtime \
    libbrotli1 \
    libjsoncpp25 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 -s /bin/bash genocr

# Set up directory structure matching build layout
RUN mkdir -p /app/api_server/build && \
    mkdir -p /app/models && \
    chown -R genocr:genocr /app

# Copy executable to build directory (maintains ../../models path)
COPY --from=builder /build/api_server/build/genocr_api_server /app/api_server/build/
COPY --from=builder /build/api_server/config/config.json /app/api_server/build/config.json

# Copy models to project root (so ../../models works from api_server/build/)
COPY --from=builder /build/models /app/models

# Create uploads directory in build
RUN mkdir -p /app/api_server/build/uploads && \
    chown -R genocr:genocr /app

# Copy ONNX Runtime libraries (standard)
COPY --from=builder /build/cpp_core/src/third_party/onnxruntime_standard/onnxruntime-linux-x64-*/lib/*.so* /usr/local/lib/

# Copy ONNX Runtime GenAI libraries
COPY --from=builder /build/cpp_core/src/third_party/onnxruntime_genai/onnxruntime-genai-*/lib/*.so* /usr/local/lib/

# Update library cache
RUN ldconfig

# Switch to non-root user
USER genocr

# Set working directory to match build environment
WORKDIR /app/api_server/build

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/api/v1/health || exit 1

# Start the server
CMD ["./genocr_api_server"]
