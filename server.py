from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from ocr import OCRNeuralNetwork


nn = OCRNeuralNetwork()


class Server(BaseHTTPRequestHandler):

    def do_POST(self):
        response_code = 200
        response = ""

        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)

        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()

        elif payload.get('predict'):
            try:
                response = {
                    "type": "test",
                    "result": nn.predict(str(payload['image']))
                }
            except Exception:
                response_code = 500

        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if response:
            self.wfile.write(json.dumps(response).encode())


server = HTTPServer(("localhost", 8000), Server)

print("OCR server running on http://localhost:8000")

server.serve_forever()