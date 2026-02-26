import 'package:flutter/material.dart';
import '../../main.dart';
import '../../services/api_service.dart';
import '../../theme/app_theme.dart';

class MealHistoryScreen extends StatefulWidget {
  const MealHistoryScreen({super.key});

  @override
  State<MealHistoryScreen> createState() => _MealHistoryScreenState();
}

class _MealHistoryScreenState extends State<MealHistoryScreen> {
  bool _loading = true;
  String? _error;
  List<Map<String, dynamic>> _conversations = [];

  // null  = currently loading; empty list = loaded (no messages / error)
  final Map<String, List<Map<String, dynamic>>?> _messages = {};

  @override
  void initState() {
    super.initState();
    _loadConversations();
  }

  Future<void> _loadConversations() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data =
          await AppServices.instance.api.get('/api/conversations') as List<dynamic>;
      if (!mounted) return;
      setState(() {
        _conversations =
            data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.message;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _error = 'Could not load history. Is the server running?';
        _loading = false;
      });
    }
  }

  Future<void> _loadMessages(String conversationId) async {
    if (_messages.containsKey(conversationId)) return; // already loaded / loading
    setState(() => _messages[conversationId] = null); // null signals loading
    try {
      final data = await AppServices.instance.api
          .get('/api/conversations/$conversationId/messages') as List<dynamic>;
      if (!mounted) return;
      setState(() {
        _messages[conversationId] =
            data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _messages[conversationId] = []);
    }
  }

  String _formatDate(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final diff = DateTime.now().difference(dt);
      if (diff.inDays == 0) {
        return 'Today ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
      }
      if (diff.inDays == 1) return 'Yesterday';
      if (diff.inDays < 7) return '${diff.inDays} days ago';
      return '${dt.day}/${dt.month}/${dt.year}';
    } catch (_) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Meal History'),
        actions: [
          IconButton(
            onPressed: _loadConversations,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _ErrorView(message: _error!, onRetry: _loadConversations)
              : _conversations.isEmpty
                  ? _EmptyView(
                      icon: Icons.history,
                      title: 'No conversations yet',
                      subtitle: 'Chat with NutriAI to get started',
                    )
                  : RefreshIndicator(
                      onRefresh: _loadConversations,
                      child: ListView.builder(
                        padding: const EdgeInsets.all(12),
                        itemCount: _conversations.length,
                        itemBuilder: (context, i) {
                          final conv = _conversations[i];
                          final id = conv['conversation_id'] as String? ?? '';
                          return _ConversationTile(
                            conversation: conv,
                            messages: _messages[id],
                            onExpand: () => _loadMessages(id),
                            formatDate: _formatDate,
                          );
                        },
                      ),
                    ),
    );
  }
}

// ─── Conversation tile with lazy-loaded messages ──────────────────────────────
class _ConversationTile extends StatelessWidget {
  final Map<String, dynamic> conversation;
  final List<Map<String, dynamic>>? messages; // null = loading
  final VoidCallback onExpand;
  final String Function(String) formatDate;

  const _ConversationTile({
    required this.conversation,
    required this.messages,
    required this.onExpand,
    required this.formatDate,
  });

  @override
  Widget build(BuildContext context) {
    final title = conversation['title'] as String? ?? 'Conversation';
    final createdAt = conversation['created_at'] as String? ?? '';
    final lastAt = conversation['last_message_at'] as String? ?? createdAt;

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ExpansionTile(
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: AppColors.cardGreen,
            borderRadius: BorderRadius.circular(10),
          ),
          child: const Icon(Icons.chat_bubble_outline,
              color: AppColors.primary, size: 20),
        ),
        title: Text(
          title,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
        ),
        subtitle: Text(
          formatDate(lastAt),
          style: const TextStyle(fontSize: 12, color: Colors.grey),
        ),
        onExpansionChanged: (expanded) {
          if (expanded) onExpand();
        },
        children: [
          if (messages == null)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
            )
          else if (messages!.isEmpty)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text('No messages found.',
                  style: TextStyle(color: Colors.grey)),
            )
          else ...[
            ...messages!.map((msg) =>
                _MessageRow(message: msg, formatDate: formatDate)),
            const SizedBox(height: 8),
          ],
        ],
      ),
    );
  }
}

// ─── Single message row inside an expanded conversation ───────────────────────
class _MessageRow extends StatelessWidget {
  final Map<String, dynamic> message;
  final String Function(String) formatDate;

  const _MessageRow({required this.message, required this.formatDate});

  @override
  Widget build(BuildContext context) {
    final role = message['role'] as String? ?? 'user';
    final content = message['content'] as String? ?? '';
    final createdAt = message['created_at'] as String? ?? '';
    final isUser = role == 'user';

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 6, 16, 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 14,
            backgroundColor:
                isUser ? AppColors.primary : AppColors.cardGreen,
            child: Icon(
              isUser ? Icons.person : Icons.eco,
              size: 14,
              color: isUser ? Colors.white : AppColors.primary,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      isUser ? 'You' : 'NutriAI',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: isUser
                            ? AppColors.primaryDark
                            : Colors.grey[600],
                      ),
                    ),
                    if (createdAt.isNotEmpty) ...[
                      const SizedBox(width: 8),
                      Text(
                        formatDate(createdAt),
                        style:
                            const TextStyle(fontSize: 10, color: Colors.grey),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  content,
                  maxLines: 5,
                  overflow: TextOverflow.ellipsis,
                  style:
                      const TextStyle(fontSize: 13, color: Colors.black87),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Shared helpers ───────────────────────────────────────────────────────────
class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.wifi_off, size: 48, color: Colors.grey),
          const SizedBox(height: 16),
          Text(message, style: const TextStyle(color: Colors.grey)),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class _EmptyView extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  const _EmptyView(
      {required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 64, color: Colors.grey[300]),
          const SizedBox(height: 16),
          Text(title,
              style: TextStyle(color: Colors.grey[500], fontSize: 16)),
          const SizedBox(height: 6),
          Text(subtitle,
              style: TextStyle(color: Colors.grey[400], fontSize: 13)),
        ],
      ),
    );
  }
}
