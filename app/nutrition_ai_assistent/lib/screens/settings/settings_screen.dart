import 'package:flutter/material.dart';
import '../../main.dart';
import '../../theme/app_theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _displayName = '';
  String _username = '';

  @override
  void initState() {
    super.initState();
    _loadAccount();
  }

  Future<void> _loadAccount() async {
    final storage = AppServices.instance.storage;
    final name = await storage.getName();
    final username = await storage.getUsername();
    if (mounted) {
      setState(() {
        _displayName = name ?? '';
        _username = username ?? '';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Account ────────────────────────────────────────────────────
            _SectionLabel('Account'),
            Card(
              child: Column(
                children: [
                  ListTile(
                    leading: const Icon(Icons.person_outline,
                        color: AppColors.primary),
                    title: const Text('Name'),
                    trailing: Text(
                      _displayName.isNotEmpty ? _displayName : '—',
                      style: const TextStyle(color: Colors.grey),
                    ),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.alternate_email,
                        color: AppColors.primary),
                    title: const Text('Username'),
                    trailing: Text(
                      _username.isNotEmpty ? '@$_username' : '—',
                      style: const TextStyle(color: Colors.grey),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── About ──────────────────────────────────────────────────────
            _SectionLabel('About'),
            Card(
              child: Column(
                children: [
                  const ListTile(
                    leading: Icon(Icons.eco, color: AppColors.primary),
                    title: Text('NutriAI'),
                    trailing: Text('v1.0.0',
                        style: TextStyle(color: Colors.grey)),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.info_outline,
                        color: AppColors.primary),
                    title: const Text('About the app'),
                    trailing: const Icon(Icons.chevron_right,
                        color: Colors.grey),
                    onTap: () => showAboutDialog(
                      context: context,
                      applicationName: 'NutriAI',
                      applicationVersion: '1.0.0',
                      applicationIcon: const Icon(Icons.eco,
                          color: AppColors.primary, size: 40),
                      children: [
                        const Text('Your Personal Nutrition AI Assistant'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Session ────────────────────────────────────────────────────
            _SectionLabel('Session'),
            Card(
              child: ListTile(
                leading: const Icon(Icons.logout, color: AppColors.error),
                title: const Text('Sign Out',
                    style: TextStyle(color: AppColors.error)),
                onTap: () async {
                  AppServices.instance.chat.disconnect();
                  await AppServices.instance.auth.logout();
                  if (!context.mounted) return;
                  Navigator.of(context).pushReplacementNamed('/login');
                },
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        text,
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
              color: AppColors.primaryDark,
              fontWeight: FontWeight.w600,
            ),
      ),
    );
  }
}
