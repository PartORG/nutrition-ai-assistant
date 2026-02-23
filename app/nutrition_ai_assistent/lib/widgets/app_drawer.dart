import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../screens/saved_recipes/saved_recipes_screen.dart';
import '../screens/settings/settings_screen.dart';

class AppDrawer extends StatelessWidget {
  final int selectedIndex;
  final String displayName;
  final String username;
  final ValueChanged<int> onItemSelected;
  final VoidCallback onSignOut;

  const AppDrawer({
    super.key,
    required this.selectedIndex,
    required this.displayName,
    required this.username,
    required this.onItemSelected,
    required this.onSignOut,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          DrawerHeader(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [AppColors.primaryDark, AppColors.primary],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const CircleAvatar(
                  radius: 30,
                  backgroundColor: Colors.white24,
                  child: Icon(Icons.person, size: 30, color: Colors.white),
                ),
                const SizedBox(height: 10),
                Text(
                  displayName.isNotEmpty ? displayName : 'NutriAI User',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (username.isNotEmpty)
                  Text(
                    '@$username',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.8),
                      fontSize: 13,
                    ),
                  ),
              ],
            ),
          ),
          _DrawerItem(
            icon: Icons.dashboard_outlined,
            label: 'Dashboard',
            selected: selectedIndex == 0,
            onTap: () { onItemSelected(0); Navigator.pop(context); },
          ),
          _DrawerItem(
            icon: Icons.chat_outlined,
            label: 'Chat with AI',
            selected: selectedIndex == 1,
            onTap: () { onItemSelected(1); Navigator.pop(context); },
          ),
          _DrawerItem(
            icon: Icons.person_outline,
            label: 'Profile',
            selected: selectedIndex == 2,
            onTap: () { onItemSelected(2); Navigator.pop(context); },
          ),
          const Divider(),
          _DrawerItem(
            icon: Icons.bookmark_outline,
            label: 'Saved Recipes',
            selected: false,
            onTap: () {
              Navigator.pop(context);
              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) => const SavedRecipesScreen()),
              );
            },
          ),
          _DrawerItem(
            icon: Icons.settings_outlined,
            label: 'Settings',
            selected: false,
            onTap: () {
              Navigator.pop(context);
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const SettingsScreen()),
              );
            },
          ),
          const Spacer(),
          const Divider(),
          _DrawerItem(
            icon: Icons.logout,
            label: 'Sign Out',
            selected: false,
            color: AppColors.error,
            onTap: () { Navigator.pop(context); onSignOut(); },
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

class _DrawerItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;
  final Color? color;

  const _DrawerItem({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveColor = color ?? (selected ? AppColors.primary : Colors.grey[700]!);
    return ListTile(
      leading: Icon(icon, color: effectiveColor),
      title: Text(
        label,
        style: TextStyle(
          color: color ?? (selected ? AppColors.primary : Colors.grey[800]),
          fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
        ),
      ),
      selected: selected,
      selectedTileColor: AppColors.cardGreen,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      onTap: onTap,
    );
  }
}
