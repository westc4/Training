# -----------------------------
# Install Claude AI
export ANTHROPIC_API_KEY="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"

apt-get update
apt-get install -y curl ca-certificates bash

curl -fsSL https://claude.ai/install.sh | bash

# Ensure installed binary is reachable (per Claude Code docs, native install uses ~/.local/bin)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# 5) Smoke test (won't print your key)
command -v claude
claude --version