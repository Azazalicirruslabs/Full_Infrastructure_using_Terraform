#!/bin/bash
# Load environment variables from .env.dev
# Usage: source ./load-env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.dev"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from .env.dev..."
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Remove carriage return (Windows line endings)
        key=$(echo "$key" | tr -d '\r')
        value=$(echo "$value" | tr -d '\r')
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Export the variable
        export "$key=$value"
        echo "  Set: $key"
    done < "$ENV_FILE"
    echo -e "\nEnvironment loaded successfully!"
else
    echo "Error: .env.dev not found at $ENV_FILE"
fi
