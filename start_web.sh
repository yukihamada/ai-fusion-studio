#!/bin/bash
echo "🚀 AI Fusion Studio 起動中..."
echo "プロフェッショナルWebインターフェースでAIモデル融合実験を開始します"
echo ""

# AI Fusion Studio専用ポート (8932 = AI + Fusion + Studio)
PORT=8932
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  ポート $PORT は既に使用されています"
    echo "別のポートで起動します..."
    PORT=8933
fi

echo "🌐 Webアプリケーションを起動..."
echo "URL: http://localhost:$PORT"
echo ""
echo "Ctrl+C で停止できます"
echo ""

cd "$(dirname "$0")"
streamlit run web/app.py --server.port $PORT --server.headless true