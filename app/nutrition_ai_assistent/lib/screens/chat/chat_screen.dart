import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:image_picker/image_picker.dart';
import '../../main.dart';
import '../../theme/app_theme.dart';
import 'camera_screen.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  final String? imagePath;

  ChatMessage({required this.text, required this.isUser, this.imagePath});
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _messageController = TextEditingController();
  final _scrollController = ScrollController();
  bool get _isCameraSupported {
    if (kIsWeb) return true;
    return Platform.isAndroid || Platform.isIOS || Platform.isWindows || Platform.isMacOS;
  }

  bool _connected = false;
  bool _connecting = false;
  bool _waitingForResponse = false;
  bool _historyLoaded = false;
  bool _uploadingImage = false;

  // Server path of an uploaded image waiting to be sent with the next message
  String? _pendingImagePath;
  String? _pendingImageName;

  final List<ChatMessage> _messages = [
    ChatMessage(
      text: 'Hello! I\'m your NutriAI assistant. How can I help you today?',
      isUser: false,
    ),
  ];

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    AppServices.instance.chat.disconnect();
    super.dispose();
  }

  Future<void> _connect() async {
    setState(() {
      _connecting = true;
      _connected = false;
    });
    try {
      await AppServices.instance.chat.connect(
        onMessage: (text) {
          if (!mounted) return;
          setState(() {
            _waitingForResponse = false;
            _messages.add(ChatMessage(text: text, isUser: false));
          });
          _scrollToBottom();
        },
        onError: (_) {
          if (mounted)
            setState(() {
              _connected = false;
              _waitingForResponse = false;
            });
        },
        onDone: () {
          if (mounted)
            setState(() {
              _connected = false;
              _waitingForResponse = false;
            });
        },
      );
      if (mounted) setState(() => _connected = true);
      if (!_historyLoaded) {
        _historyLoaded = true;
        await _loadHistory();
      }
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _connected = false;
        _messages.add(
          ChatMessage(
            text:
                'Could not connect to server. Make sure the API is running and you are logged in.',
            isUser: false,
          ),
        );
      });
    } finally {
      if (mounted) setState(() => _connecting = false);
    }
  }

  Future<void> _loadHistory() async {
    try {
      final data = await AppServices.instance.api.get(
        '/api/chat/history?hours=24',
      );
      final rawMessages = data['messages'] as List<dynamic>? ?? [];
      if (!mounted || rawMessages.isEmpty) return;

      final history = rawMessages
          .map(
            (m) => ChatMessage(
              text: (m['content'] as String?) ?? '',
              isUser: (m['role'] as String?) == 'user',
            ),
          )
          .toList();

      setState(() {
        _messages
          ..clear()
          ..addAll(history);
      });
      _scrollToBottom();
    } catch (_) {
      // Best-effort — silently ignore if history is unavailable
    }
  }

  void _sendMessage() {
    final text = _messageController.text.trim();
    final hasPendingImage = _pendingImagePath != null;

    if (text.isEmpty && !hasPendingImage) return;
    if (!_connected) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Not connected. Tap the reconnect button.'),
        ),
      );
      return;
    }

    final String wsMessage;
    final String displayText;
    final String? displayImagePath;

    if (hasPendingImage) {
      // Combine user text + image reference for the backend agent
      wsMessage = text.isNotEmpty
          ? '$text\n[IMAGE:$_pendingImagePath]'
          : '[IMAGE:$_pendingImagePath]';
      displayText = text.isNotEmpty
          ? text
          : '📷 ${_pendingImageName ?? 'Photo'}';
      displayImagePath = _pendingImagePath;
    } else {
      wsMessage = text;
      displayText = text;
      displayImagePath = null;
    }

    setState(() {
      _messages.add(
        ChatMessage(
          text: displayText,
          isUser: true,
          imagePath: displayImagePath,
        ),
      );
      _messageController.clear();
      _pendingImagePath = null;
      _pendingImageName = null;
      _waitingForResponse = true;
    });
    AppServices.instance.chat.send(wsMessage);
    _scrollToBottom();
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showImageOptions() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Add Food Photo',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  color: AppColors.primaryDark,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Take a photo or choose from your library to analyze food',
                style: Theme.of(
                  context,
                ).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              if (_isCameraSupported) ...[
                ListTile(
                  leading: Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppColors.cardGreen,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.camera_alt,
                      color: AppColors.primary,
                    ),
                  ),
                  title: const Text('Take Photo'),
                  subtitle: const Text('Use camera to capture food'),
                  onTap: _openCamera,
                ),
                const SizedBox(height: 8),
              ],
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppColors.cardGreen,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.photo_library,
                    color: AppColors.primary,
                  ),
                ),
                title: const Text('Choose from Library'),
                subtitle: const Text('Select an existing photo'),
                onTap: () => _pickImage(ImageSource.gallery),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  /// Upload [image] to the server and stage it for the next message.
  Future<void> _uploadImage(XFile image) async {
    if (!_connected) return;
    setState(() => _uploadingImage = true);
    try {
      final bytes = await image.readAsBytes();
      final fileName = image.name.isNotEmpty
          ? image.name
          : 'photo_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final serverPath = await AppServices.instance.api.uploadImageBytes(
        bytes,
        fileName,
      );
      if (!mounted) return;
      setState(() {
        _uploadingImage = false;
        _pendingImagePath = serverPath;
        _pendingImageName = fileName;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _uploadingImage = false;
        _messages.add(
          ChatMessage(text: 'Could not upload image: $e', isUser: false),
        );
      });
    }
  }

  /// Pick a photo from the gallery (all platforms).
  Future<void> _pickImage(ImageSource source) async {
    Navigator.of(context).pop();
    final XFile? image = await ImagePicker().pickImage(source: source);
    if (image == null) return;
    await _uploadImage(image);
  }

  /// Open the live-camera screen (web + Windows + mobile) and capture a photo.
  Future<void> _openCamera() async {
    Navigator.of(context).pop();
    final XFile? image = await Navigator.push<XFile?>(
      context,
      MaterialPageRoute(builder: (_) => const CameraScreen()),
    );
    if (image == null) return;
    await _uploadImage(image);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (_connecting)
          const LinearProgressIndicator(
            backgroundColor: AppColors.cardGreen,
            color: AppColors.primary,
          ),
        if (!_connected && !_connecting)
          Container(
            color: Colors.orange.shade50,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                const Icon(Icons.wifi_off, color: Colors.orange, size: 18),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Disconnected',
                    style: TextStyle(color: Colors.orange, fontSize: 13),
                  ),
                ),
                TextButton(
                  onPressed: _connect,
                  child: const Text(
                    'Reconnect',
                    style: TextStyle(fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
        Expanded(
          child: ListView.builder(
            controller: _scrollController,
            padding: const EdgeInsets.all(16),
            itemCount: _messages.length + (_waitingForResponse ? 1 : 0),
            itemBuilder: (context, index) {
              if (_waitingForResponse && index == _messages.length) {
                return const _TypingIndicator();
              }
              return _MessageBubble(message: _messages[index]);
            },
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: Colors.white,
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 10,
                offset: const Offset(0, -2),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: SafeArea(
            top: false,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // ── Upload progress indicator ──────────────────────────────
                if (_uploadingImage)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      children: [
                        const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: AppColors.primary,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Uploading photo…',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ),

                // ── Pending image chip ─────────────────────────────────────
                if (_pendingImagePath != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 5,
                          ),
                          decoration: BoxDecoration(
                            color: AppColors.cardGreen,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(
                                Icons.image_outlined,
                                size: 16,
                                color: AppColors.primary,
                              ),
                              const SizedBox(width: 6),
                              Text(
                                _pendingImageName ?? 'Photo attached',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: AppColors.primaryDark,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 4),
                        GestureDetector(
                          onTap: () => setState(() {
                            _pendingImagePath = null;
                            _pendingImageName = null;
                          }),
                          child: const Icon(
                            Icons.close,
                            size: 16,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                  ),

                // ── Text field row ─────────────────────────────────────────
                Row(
                  children: [
                    IconButton(
                      onPressed: (_uploadingImage || _waitingForResponse)
                          ? null
                          : _showImageOptions,
                      icon: const Icon(Icons.camera_alt_outlined),
                      color: AppColors.primary,
                      tooltip: 'Add photo',
                    ),
                    Expanded(
                      child: TextField(
                        controller: _messageController,
                        enabled: !_waitingForResponse && !_uploadingImage,
                        decoration: InputDecoration(
                          hintText: _pendingImagePath != null
                              ? 'Add a message for this photo…'
                              : 'Ask about nutrition…',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide.none,
                          ),
                          filled: true,
                          fillColor: AppColors.cardGreen,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 10,
                          ),
                        ),
                        textInputAction: TextInputAction.send,
                        onSubmitted: (_) => _sendMessage(),
                      ),
                    ),
                    const SizedBox(width: 4),
                    CircleAvatar(
                      backgroundColor:
                          (_connected &&
                              !_waitingForResponse &&
                              !_uploadingImage)
                          ? AppColors.primary
                          : Colors.grey,
                      child: IconButton(
                        onPressed:
                            (_connected &&
                                !_waitingForResponse &&
                                !_uploadingImage)
                            ? _sendMessage
                            : null,
                        icon: const Icon(
                          Icons.send,
                          color: Colors.white,
                          size: 20,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(right: 8, top: 4),
            padding: const EdgeInsets.all(6),
            decoration: const BoxDecoration(
              color: AppColors.cardGreen,
              shape: BoxShape.circle,
            ),
            child: Image.asset(
              'assets/icons/icon_chat.png',
              width: 28,
              height: 28,
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(16),
                topRight: Radius.circular(16),
                bottomLeft: Radius.circular(4),
                bottomRight: Radius.circular(16),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 5,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                return AnimatedBuilder(
                  animation: _controller,
                  builder: (_, __) {
                    final phase = (_controller.value - i * 0.2).clamp(0.0, 1.0);
                    final opacity =
                        (0.3 +
                                0.7 *
                                    (phase < 0.5
                                        ? phase / 0.5
                                        : (1.0 - phase) / 0.5))
                            .clamp(0.3, 1.0);
                    return Container(
                      margin: EdgeInsets.only(right: i < 2 ? 4 : 0),
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: AppColors.primary.withValues(alpha: opacity),
                        shape: BoxShape.circle,
                      ),
                    );
                  },
                );
              }),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final ChatMessage message;
  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser)
            Container(
              margin: const EdgeInsets.only(right: 8, top: 4),
              padding: const EdgeInsets.all(6),
              decoration: const BoxDecoration(
                color: AppColors.cardGreen,
                shape: BoxShape.circle,
              ),
              child: Image.asset(
                'assets/icons/icon_chat.png',
                width: 28,
                height: 28,
              ),
            ),
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser ? AppColors.primary : Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(16),
                  topRight: const Radius.circular(16),
                  bottomLeft: Radius.circular(isUser ? 16 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 16),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 5,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: isUser
                  ? Text(
                      message.text,
                      style: const TextStyle(color: Colors.white, fontSize: 14),
                    )
                  : MarkdownBody(
                      data: message.text,
                      styleSheet: MarkdownStyleSheet(
                        p: const TextStyle(color: Colors.black87, fontSize: 14),
                        strong: const TextStyle(
                          color: Colors.black87,
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                        ),
                        h1: const TextStyle(
                          color: Colors.black87,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                        h2: const TextStyle(
                          color: Colors.black87,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                        h3: const TextStyle(
                          color: Colors.black87,
                          fontSize: 15,
                          fontWeight: FontWeight.bold,
                        ),
                        listBullet: const TextStyle(
                          color: Colors.black87,
                          fontSize: 14,
                        ),
                        blockquote: const TextStyle(
                          color: Colors.black54,
                          fontSize: 14,
                        ),
                        code: const TextStyle(
                          fontSize: 13,
                          backgroundColor: Color(0xFFF0F0F0),
                        ),
                        horizontalRuleDecoration: const BoxDecoration(
                          border: Border(
                            top: BorderSide(color: Color(0xFFE0E0E0)),
                          ),
                        ),
                      ),
                      shrinkWrap: true,
                    ),
            ),
          ),
          if (isUser)
            Container(
              margin: const EdgeInsets.only(left: 8, top: 4),
              padding: const EdgeInsets.all(6),
              decoration: const BoxDecoration(
                color: AppColors.primary,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.person, size: 18, color: Colors.white),
            ),
        ],
      ),
    );
  }
}
