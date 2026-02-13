FROM rust:1.93-bullseye

WORKDIR /data

COPY data/ .

RUN cargo build --release

EXPOSE 80

CMD ["./target/release/proxy"]