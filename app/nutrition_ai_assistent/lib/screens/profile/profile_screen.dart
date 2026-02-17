import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _buildProfileHeader(context),
          const SizedBox(height: 16),
          _buildInfoSection(context),
          const SizedBox(height: 16),
          _buildHealthSection(context),
          const SizedBox(height: 16),
          _buildPreferencesSection(context),
          const SizedBox(height: 16),
          _buildActionsSection(context),
        ],
      ),
    );
  }

  Widget _buildProfileHeader(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Stack(
              alignment: Alignment.bottomRight,
              children: [
                CircleAvatar(
                  radius: 48,
                  backgroundColor: AppColors.cardGreen,
                  child: const Icon(Icons.person, size: 48, color: AppColors.primary),
                ),
                Container(
                  padding: const EdgeInsets.all(4),
                  decoration: const BoxDecoration(
                    color: AppColors.primary,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.edit, size: 16, color: Colors.white),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'John Doe',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '@johndoe',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Colors.grey,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _StatItem(label: 'Recipes', value: '24'),
                Container(height: 30, width: 1, color: Colors.grey[300]),
                _StatItem(label: 'Days Active', value: '15'),
                Container(height: 30, width: 1, color: Colors.grey[300]),
                _StatItem(label: 'Goals Met', value: '8'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoSection(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Personal Information',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  onPressed: () {
                    // TODO: Edit personal info
                  },
                  icon: const Icon(Icons.edit_outlined, size: 20),
                  color: AppColors.primary,
                ),
              ],
            ),
            const _InfoRow(icon: Icons.person_outline, label: 'Name', value: 'John Doe'),
            const _InfoRow(icon: Icons.cake_outlined, label: 'Age', value: '28 years'),
            const _InfoRow(icon: Icons.wc, label: 'Gender', value: 'Male'),
            const _InfoRow(icon: Icons.monitor_weight_outlined, label: 'Weight', value: '75 kg'),
            const _InfoRow(icon: Icons.height, label: 'Height', value: '178 cm'),
          ],
        ),
      ),
    );
  }

  Widget _buildHealthSection(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Health Conditions',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  onPressed: () {
                    // TODO: Edit health conditions
                  },
                  icon: const Icon(Icons.edit_outlined, size: 20),
                  color: AppColors.primary,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _ConditionChip(label: 'Diabetes Type 2'),
                _ConditionChip(label: 'Hypertension'),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.cardGreen,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(Icons.medical_services_outlined,
                      color: AppColors.primary, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Your dietary recommendations are personalized based on your health conditions.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreferencesSection(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Dietary Preferences',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  onPressed: () {
                    // TODO: Edit preferences
                  },
                  icon: const Icon(Icons.edit_outlined, size: 20),
                  color: AppColors.primary,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _PreferenceChip(label: 'Low Sodium', icon: Icons.remove_circle_outline),
                _PreferenceChip(label: 'High Protein', icon: Icons.fitness_center),
                _PreferenceChip(label: 'Mediterranean', icon: Icons.restaurant),
                _PreferenceChip(label: 'No Pork', icon: Icons.block),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionsSection(BuildContext context) {
    return Column(
      children: [
        Card(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.notifications_outlined, color: AppColors.primary),
                title: const Text('Notifications'),
                trailing: Switch(
                  value: true,
                  onChanged: (val) {
                    // TODO: Toggle notifications
                  },
                  activeThumbColor: AppColors.primary,
                ),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.language, color: AppColors.primary),
                title: const Text('Language'),
                trailing: const Text('English', style: TextStyle(color: Colors.grey)),
                onTap: () {
                  // TODO: Language selection
                },
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.dark_mode_outlined, color: AppColors.primary),
                title: const Text('Dark Mode'),
                trailing: Switch(
                  value: false,
                  onChanged: (val) {
                    // TODO: Toggle dark mode
                  },
                  activeThumbColor: AppColors.primary,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () {
              // TODO: Logout
              Navigator.of(context).pushReplacementNamed('/login');
            },
            icon: const Icon(Icons.logout, color: AppColors.error),
            label: const Text('Sign Out', style: TextStyle(color: AppColors.error)),
            style: OutlinedButton.styleFrom(
              side: const BorderSide(color: AppColors.error),
              padding: const EdgeInsets.symmetric(vertical: 14),
            ),
          ),
        ),
        const SizedBox(height: 24),
      ],
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;

  const _StatItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
            color: AppColors.primary,
          ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.grey,
          ),
        ),
      ],
    );
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _InfoRow({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Icon(icon, size: 20, color: AppColors.primary),
          const SizedBox(width: 12),
          SizedBox(
            width: 70,
            child: Text(label, style: const TextStyle(color: Colors.grey)),
          ),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}

class _ConditionChip extends StatelessWidget {
  final String label;

  const _ConditionChip({required this.label});

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label, style: const TextStyle(color: AppColors.primaryDark, fontSize: 13)),
      backgroundColor: AppColors.cardGreen,
      side: BorderSide.none,
      avatar: const Icon(Icons.medical_services, size: 16, color: AppColors.primary),
    );
  }
}

class _PreferenceChip extends StatelessWidget {
  final String label;
  final IconData icon;

  const _PreferenceChip({required this.label, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label, style: const TextStyle(color: AppColors.primaryDark, fontSize: 13)),
      backgroundColor: AppColors.cardGreen,
      side: BorderSide.none,
      avatar: Icon(icon, size: 16, color: AppColors.primary),
    );
  }
}
