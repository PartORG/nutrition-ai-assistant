import 'package:flutter/material.dart';
import '../../main.dart';
import '../../services/api_service.dart';
import '../../theme/app_theme.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _loading = true;
  String? _error;

  String _displayName = '';
  String _username = '';
  int _age = 0;
  String _gender = '';
  String _caretaker = '';
  List<String> _healthConditions = [];
  List<String> _preferences = [];
  List<String> _restrictions = [];
  List<Map<String, dynamic>> _medicalAdvice = [];

  // Edit mode states
  bool _editingHeader = false;
  bool _editingHealth = false;
  bool _editingDietary = false;
  bool _editingMedical = false;

  // Edit controllers
  late TextEditingController _nameController;
  late TextEditingController _ageController;
  late TextEditingController _genderController;
  late TextEditingController _caretakerController;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController();
    _ageController = TextEditingController();
    _genderController = TextEditingController();
    _caretakerController = TextEditingController();
    _loadProfile();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _genderController.dispose();
    _caretakerController.dispose();
    super.dispose();
  }

  Future<void> _loadProfile() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = AppServices.instance.api;
      final profileData = await api.get('/api/profile') as Map<String, dynamic>;

      if (!mounted) return;
      setState(() {
        // --- User demographics from the users table ---
        final userMap = profileData['user'] as Map<String, dynamic>?;
        if (userMap != null) {
          final firstName = (userMap['name'] as String? ?? '').trim();
          final lastName = (userMap['surname'] as String? ?? '').trim();
          _displayName = [
            firstName,
            lastName,
          ].where((s) => s.isNotEmpty).join(' ');
          _username = (userMap['user_name'] as String? ?? '').trim();
          _age = userMap['age'] as int? ?? 0;
          _gender = (userMap['gender'] as String? ?? '').trim();
          _caretaker = (userMap['caretaker'] as String? ?? '').trim();
        }
        if (_displayName.isEmpty)
          _displayName = _username.isNotEmpty ? _username : 'User';

        // --- Latest dietary profile snapshot ---
        final profiles = profileData['profiles'] as List? ?? [];
        if (profiles.isNotEmpty) {
          final latest = profiles.first as Map<String, dynamic>;
          _healthConditions = _split(
            latest['health_condition'] as String? ?? '',
          );
          _preferences = _split(latest['preferences'] as String? ?? '');
          _restrictions = _split(latest['restrictions'] as String? ?? '');
        }

        _medicalAdvice = List<Map<String, dynamic>>.from(
          (profileData['medical_advice'] as List? ?? []).map(
            (m) => Map<String, dynamic>.from(m as Map),
          ),
        );
        
        _initializeControllers();
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
        _error = 'Could not load profile. Is the server running?';
        _loading = false;
      });
    }
  }

  void _initializeControllers() {
    _nameController.text = _displayName;
    _ageController.text = _age > 0 ? _age.toString() : '';
    _genderController.text = _gender;
    _caretakerController.text = _caretaker;
  }

  Future<void> _saveHeaderChanges() async {
    try {
      final updatedData = {
        'name': _nameController.text,
        'age': int.tryParse(_ageController.text) ?? 0,
        'gender': _genderController.text,
        'caretaker': _caretakerController.text,
      };

      await AppServices.instance.api.post(
        '/api/profile/update',
        updatedData,
        auth: true,
      );

      if (!mounted) return;
      setState(() {
        _displayName = _nameController.text;
        _age = int.tryParse(_ageController.text) ?? 0;
        _gender = _genderController.text;
        _caretaker = _caretakerController.text;
        _editingHeader = false;
      });

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Profile updated successfully')),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving profile: $e')),
      );
    }
  }

  void _cancelHeaderEdit() {
    setState(() {
      _editingHeader = false;
      _initializeControllers();
    });
  }

  List<String> _split(String value) => value
      .split(',')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty && s != 'None')
      .toList();

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Center(child: CircularProgressIndicator());

    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.wifi_off, size: 48, color: Colors.grey),
            const SizedBox(height: 16),
            Text(_error!, style: const TextStyle(color: Colors.grey)),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loadProfile,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadProfile,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildHeader(context),
            const SizedBox(height: 16),
            if (_healthConditions.isNotEmpty) ...[
              _buildHealthSection(context),
              const SizedBox(height: 16),
            ],
            if (_preferences.isNotEmpty || _restrictions.isNotEmpty) ...[
              _buildDietarySection(context),
              const SizedBox(height: 16),
            ],
            if (_medicalAdvice.isNotEmpty) ...[
              _buildMedicalAdviceSection(context),
              const SizedBox(height: 16),
            ],
            _buildActionsSection(context),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    final demoBadges = <String>[
      if (_age > 0) '$_age yrs',
      if (_gender.isNotEmpty) _gender,
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Spacer(),
                if (!_editingHeader)
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _editingHeader = true;
                      });
                    },
                    icon: const Icon(Icons.edit, color: AppColors.primary),
                    tooltip: 'Edit Profile',
                  ),
              ],
            ),
            const SizedBox(height: 8),
            const CircleAvatar(
              radius: 48,
              backgroundColor: AppColors.cardGreen,
              child: Icon(Icons.person, size: 48, color: AppColors.primary),
            ),
            const SizedBox(height: 12),
            if (!_editingHeader) ...[
              Text(
                _displayName,
                style: Theme.of(
                  context,
                ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
              ),
              if (_username.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  '@$_username',
                  style: Theme.of(
                    context,
                  ).textTheme.bodyMedium?.copyWith(color: Colors.grey),
                ),
              ],
              if (demoBadges.isNotEmpty) ...[
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: demoBadges
                      .map(
                        (label) => Chip(
                          label: Text(
                            label,
                            style: const TextStyle(
                              fontSize: 12,
                              color: AppColors.primaryDark,
                            ),
                          ),
                          backgroundColor: AppColors.cardGreen,
                          side: BorderSide.none,
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                          materialTapTargetSize:
                              MaterialTapTargetSize.shrinkWrap,
                        ),
                      )
                      .toList(),
                ),
              ],
              if (_caretaker.isNotEmpty) ...[
                const SizedBox(height: 10),
                const Divider(height: 1),
                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(
                      Icons.people_outline,
                      size: 16,
                      color: Colors.grey,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      'Caretaker: $_caretaker',
                      style: Theme.of(
                        context,
                      ).textTheme.bodySmall?.copyWith(color: Colors.grey),
                    ),
                  ],
                ),
              ],
            ] else ...[
              // Edit mode
              TextField(
                controller: _nameController,
                decoration: InputDecoration(
                  labelText: 'Full Name',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 12,
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    flex: 1,
                    child: TextField(
                      controller: _ageController,
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(
                        labelText: 'Age',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 12,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 1,
                    child: TextField(
                      controller: _genderController,
                      decoration: InputDecoration(
                        labelText: 'Gender',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 12,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _caretakerController,
                decoration: InputDecoration(
                  labelText: 'Caretaker',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 12,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _saveHeaderChanges,
                      icon: const Icon(Icons.check),
                      label: const Text('Save'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _cancelHeaderEdit,
                      icon: const Icon(Icons.close),
                      label: const Text('Cancel'),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                ],
              ),
            ],
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
                  style: Theme.of(
                    context,
                  ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                ),
                if (!_editingHealth)
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _editingHealth = true;
                      });
                    },
                    icon: const Icon(Icons.edit, size: 20, color: AppColors.primary),
                    tooltip: 'Edit',
                    constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                    padding: EdgeInsets.zero,
                  ),
              ],
            ),
            const SizedBox(height: 8),
            if (!_editingHealth) ...[
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _healthConditions
                    .map(
                      (c) => Chip(
                        label: Text(
                          c,
                          style: const TextStyle(
                            color: AppColors.primaryDark,
                            fontSize: 13,
                          ),
                        ),
                        backgroundColor: AppColors.cardGreen,
                        side: BorderSide.none,
                        avatar: const Icon(
                          Icons.medical_services,
                          size: 16,
                          color: AppColors.primary,
                        ),
                      ),
                    )
                    .toList(),
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
                    const Icon(
                      Icons.medical_services_outlined,
                      color: AppColors.primary,
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Your dietary recommendations are personalised based on your health conditions.',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ] else ...[
              Text(
                'Edit mode coming soon',
                style: TextStyle(color: Colors.grey[600]),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () {
                  setState(() {
                    _editingHealth = false;
                  });
                },
                icon: const Icon(Icons.close),
                label: const Text('Close'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDietarySection(BuildContext context) {
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
                  'Dietary Preferences & Restrictions',
                  style: Theme.of(
                    context,
                  ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                ),
                if (!_editingDietary)
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _editingDietary = true;
                      });
                    },
                    icon: const Icon(Icons.edit, size: 20, color: AppColors.primary),
                    tooltip: 'Edit',
                    constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                    padding: EdgeInsets.zero,
                  ),
              ],
            ),
            const SizedBox(height: 8),
            if (!_editingDietary) ...[
              if (_preferences.isNotEmpty) ...[
                Text(
                  'Preferences',
                  style: TextStyle(color: Colors.grey[600], fontSize: 13),
                ),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _preferences
                      .map(
                        (p) => Chip(
                          label: Text(
                            p,
                            style: const TextStyle(
                              color: AppColors.primaryDark,
                              fontSize: 13,
                            ),
                          ),
                          backgroundColor: AppColors.cardGreen,
                          side: BorderSide.none,
                          avatar: const Icon(
                            Icons.restaurant,
                            size: 16,
                            color: AppColors.primary,
                          ),
                        ),
                      )
                      .toList(),
                ),
              ],
              if (_restrictions.isNotEmpty) ...[
                const SizedBox(height: 12),
                Text(
                  'Restrictions',
                  style: TextStyle(color: Colors.grey[600], fontSize: 13),
                ),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _restrictions
                      .map(
                        (r) => Chip(
                          label: Text(
                            r,
                            style: const TextStyle(
                              color: AppColors.primaryDark,
                              fontSize: 13,
                            ),
                          ),
                          backgroundColor: Colors.orange.shade50,
                          side: BorderSide.none,
                          avatar: const Icon(
                            Icons.block,
                            size: 16,
                            color: Colors.orange,
                          ),
                        ),
                      )
                      .toList(),
                ),
              ],
            ] else ...[
              Text(
                'Edit mode coming soon',
                style: TextStyle(color: Colors.grey[600]),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () {
                  setState(() {
                    _editingDietary = false;
                  });
                },
                icon: const Icon(Icons.close),
                label: const Text('Close'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildMedicalAdviceSection(BuildContext context) {
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
                  'Medical Advice',
                  style: Theme.of(
                    context,
                  ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                ),
                if (!_editingMedical)
                  IconButton(
                    onPressed: () {
                      setState(() {
                        _editingMedical = true;
                      });
                    },
                    icon: const Icon(Icons.edit, size: 20, color: AppColors.primary),
                    tooltip: 'Edit',
                    constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                    padding: EdgeInsets.zero,
                  ),
              ],
            ),
            const SizedBox(height: 8),
            if (!_editingMedical) ...[
              ..._medicalAdvice.take(3).map((advice) {
                final condition = advice['health_condition'] as String? ?? '';
                final text = advice['medical_advice'] as String? ?? '';
                final avoid = advice['avoid'] as String? ?? '';
                final dietaryLimit = advice['dietary_limit'] as String? ?? '';
                return Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppColors.cardGreen,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (condition.isNotEmpty)
                        Text(
                          condition,
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            color: AppColors.primaryDark,
                          ),
                        ),
                      if (text.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          text,
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ],
                      if (dietaryLimit.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          'Limit: $dietaryLimit',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.blue[700],
                          ),
                        ),
                      ],
                      if (avoid.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          'Avoid: $avoid',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.orange[700],
                          ),
                        ),
                      ],
                    ],
                  ),
                );
              }),
            ] else ...[
              Text(
                'Edit mode coming soon',
                style: TextStyle(color: Colors.grey[600]),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () {
                  setState(() {
                    _editingMedical = false;
                  });
                },
                icon: const Icon(Icons.close),
                label: const Text('Close'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildActionsSection(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () async {
              AppServices.instance.chat.disconnect();
              await AppServices.instance.auth.logout();
              if (!context.mounted) return;
              Navigator.of(context).pushReplacementNamed('/login');
            },
            icon: const Icon(Icons.logout, color: AppColors.error),
            label: const Text(
              'Sign Out',
              style: TextStyle(color: AppColors.error),
            ),
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
}final profileData = await api.get('/api/profile') as Map<String, dynamic>;