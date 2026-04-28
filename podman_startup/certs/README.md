# Build Certificates

Place the company TLS inspection certificate here before building behind the corporate proxy.

Expected default path:

```text
podman_startup/certs/NG-Certificate-Chain.cer
```

The build script reads this file, base64-encodes it, and passes it into the image build as a build argument. The certificate file itself is ignored by git.
