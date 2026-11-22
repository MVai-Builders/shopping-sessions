FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3 -C codegen-units=1"
RUN cargo build --release

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
WORKDIR /app
COPY --from=builder /app/target/release/shopping-sessions /app/shopping-sessions
RUN chmod +x /app/shopping-sessions
USER 1000
ENTRYPOINT ["/app/shopping-sessions"]
