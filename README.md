# Mula(n)

Use HeartMuLa (3B happy-new-year) on Apple silicon.

```
# download required models/configs
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
hf download --local-dir './ckpt/HeartCodec-oss' HeartMuLa/HeartCodec-oss-20260123

# convert codec and model to MLX
uv run python convert_heartcodec.py
uv run python convert_heartmula.py

# copy HeartMuLaGen configs
cp ckpt/gen_config.json ckpt-mlx/gen_config.json
cp ckpt/tokenizer.json ckpt-mlx/tokenizer.json

# build frontend
cd web-ui && bun install && bun run build && cd ..

# start server (will serve frontend)
uv run server.py

# Navigate to http://localhost:8080
```

Really sensible to parameters, can take a lot of time if not "known".

Have fun!