import 'package:web_socket_channel/web_socket_channel.dart';
import 'api_service.dart';
import 'storage_service.dart';

/// Manages the WebSocket connection to `/ws/chat?token=...`.
class ChatWsService {
  final StorageService _storage;
  WebSocketChannel? _channel;

  ChatWsService(this._storage);

  Future<void> connect({
    required void Function(String) onMessage,
    required void Function(dynamic) onError,
    required void Function() onDone,
  }) async {
    final token = await _storage.getToken();
    if (token == null) throw Exception('Not authenticated');

    final wsBase = ApiService.baseUrl
        .replaceFirst('https://', 'wss://')
        .replaceFirst('http://', 'ws://');

    _channel = WebSocketChannel.connect(
      Uri.parse('$wsBase/ws/chat?token=$token'),
    );

    // Wait for the handshake to complete (web_socket_channel v3+).
    // Throws if the connection is refused or the server rejects it.
    await _channel!.ready;

    _channel!.stream.listen(
      (data) => onMessage(data.toString()),
      onError: onError,
      onDone: onDone,
      cancelOnError: false,
    );
  }

  void send(String message) {
    _channel?.sink.add(message);
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }

  bool get isConnected => _channel != null;
}
