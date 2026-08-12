import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';
import 'api_service.dart';
import 'storage_service.dart';

/// A frame received from the chat WebSocket.
///
/// The server previously sent bare text for both answers and failures, so an
/// error was indistinguishable from assistant speech and got rendered as if
/// the assistant had said it. Every frame now carries a type.
sealed class ChatFrame {
  const ChatFrame();

  /// Parse a server frame. Text that is not a known frame is treated as an
  /// answer, so an older server does not break the client outright.
  static ChatFrame parse(String raw) {
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! Map<String, dynamic>) return ChatAnswer(raw);

      switch (decoded['type']) {
        case 'answer':
          return ChatAnswer(
            (decoded['content'] ?? '') as String,
            conversationId: decoded['conversation_id'] as String?,
          );
        case 'status':
          return ChatStatus(
            (decoded['stage'] ?? '') as String,
            (decoded['detail'] ?? '') as String,
          );
        case 'error':
          return ChatError(
            (decoded['message'] ?? 'Something went wrong.') as String,
            recoverable: (decoded['recoverable'] ?? true) as bool,
          );
        default:
          return ChatAnswer(raw);
      }
    } on FormatException {
      return ChatAnswer(raw);
    }
  }
}

/// The assistant's reply, to be shown in the conversation.
class ChatAnswer extends ChatFrame {
  final String content;
  final String? conversationId;
  const ChatAnswer(this.content, {this.conversationId});
}

/// Progress while a turn is running. A turn can take tens of seconds against a
/// local model, so this is what keeps the UI from looking hung.
class ChatStatus extends ChatFrame {
  final String stage;
  final String detail;
  const ChatStatus(this.stage, this.detail);
}

/// A failure. Must be rendered as an error, never as assistant speech.
class ChatError extends ChatFrame {
  final String message;
  final bool recoverable;
  const ChatError(this.message, {this.recoverable = true});
}

/// Manages the WebSocket connection to `/ws/chat?token=...`.
class ChatWsService {
  final StorageService _storage;
  WebSocketChannel? _channel;

  ChatWsService(this._storage);

  Future<void> connect({
    required void Function(ChatFrame) onFrame,
    required void Function(dynamic) onError,
    required void Function() onDone,
  }) async {
    final token = await _storage.getToken();
    if (token == null) throw Exception('Not authenticated');

    final wsBase = ApiService.baseUrl
        .replaceFirst('https://', 'wss://')
        .replaceFirst('http://', 'ws://');

    _channel = WebSocketChannel.connect(
      Uri.parse('$wsBase/api/ws/chat?token=$token'),
    );

    // Wait for the handshake to complete (web_socket_channel v3+).
    // Throws if the connection is refused or the server rejects it.
    await _channel!.ready;

    _channel!.stream.listen(
      (data) => onFrame(ChatFrame.parse(data.toString())),
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
