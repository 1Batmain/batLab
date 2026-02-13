use axum::Router;
use axum_reverse_proxy::ReverseProxy;

fn frontend_url() -> String {
    std::env::var("FRONTEND_URL").unwrap_or_else(|_| "http://127.0.0.1:5173".into())
}

#[tokio::main]
async fn main() {
    let url = frontend_url();
    println!("proxy listening on :80 → {url}");

    let proxy = ReverseProxy::new("/", &url);
    let app: Router = proxy.into();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:80").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}