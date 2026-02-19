import 'package:flutter/material.dart';
import '../../main.dart';
import '../../widgets/app_drawer.dart';
import '../dashboard/dashboard_screen.dart';
import '../chat/chat_screen.dart';
import '../profile/profile_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  String _username = '';
  String _displayName = '';

  @override
  void initState() {
    super.initState();
    _loadUserInfo();
  }

  Future<void> _loadUserInfo() async {
    final storage = AppServices.instance.storage;
    final username = await storage.getUsername();
    final name = await storage.getName();
    if (mounted) {
      setState(() {
        _username = username ?? '';
        _displayName = name ?? username ?? '';
      });
    }
  }

  final List<String> _titles = [
    'Dashboard',
    'Chat with NutriAI',
    'Profile',
  ];

  Future<void> _handleSignOut() async {
    AppServices.instance.chat.disconnect();
    await AppServices.instance.auth.logout();
    if (!mounted) return;
    Navigator.of(context).pushReplacementNamed('/login');
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      const DashboardScreen(),
      const ChatScreen(),
      const ProfileScreen(),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(_titles[_currentIndex]),
        actions: [
          if (_currentIndex == 1)
            IconButton(
              onPressed: () {
                AppServices.instance.chat.disconnect();
                setState(() {});
              },
              icon: const Icon(Icons.add_comment_outlined),
              tooltip: 'New Chat',
            ),
        ],
      ),
      drawer: AppDrawer(
        selectedIndex: _currentIndex,
        displayName: _displayName,
        username: _username,
        onItemSelected: (index) {
          setState(() => _currentIndex = index);
        },
        onSignOut: _handleSignOut,
      ),
      body: screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() => _currentIndex = index);
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard_outlined),
            activeIcon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.chat_outlined),
            activeIcon: Icon(Icons.chat),
            label: 'Chat',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            activeIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}
