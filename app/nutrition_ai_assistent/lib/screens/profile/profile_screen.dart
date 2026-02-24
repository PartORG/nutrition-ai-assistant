import 'dart:convert';

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
  bool _editingConstraints = false;

  // Edit controllers
  late TextEditingController _nameController;
  late TextEditingController _ageController;
  late TextEditingController _genderController;
  late TextEditingController _caretakerController;
  late TextEditingController _healthController;
  late TextEditingController _constraintsController;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController();
    _ageController = TextEditingController();
    _genderController = TextEditingController();
    _caretakerController = TextEditingController();
    _healthController = TextEditingController();
    _constraintsController = TextEditingController();
    _loadProfile();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _genderController.dispose();
    _caretakerController.dispose();
    _healthController.dispose();
    _constraintsController.dispose();
    super.dispose();
  }

  Future<void> _loadProfile() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = AppServices.instance.api;
      final profileData = await api.get('/api') as Map<String, dynamic>;

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
    _healthController.text = _healthConditions.join(', ');
    _constraintsController.text = _medicalAdvice
        .where((a) => (a['dietary_constraints'] as String? ?? '').isNotEmpty)
        .map((a) => a['dietary_constraints'] as String)
        .join(', ');
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
        '/api/update',
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

  Future<void> _saveHealthChanges() async {
    try {
      await AppServices.instance.api.post(
        '/api/update-health',
        {'health_condition': _healthController.text},
        auth: true,
      );

      if (!mounted) return;
      setState(() {
        _healthConditions = _healthController.text
            .split(',')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .toList();
        _editingHealth = false;
      });

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Health conditions updated successfully')),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving health conditions: $e')),
      );
    }
  }

  Future<void> _saveConstraintsChanges() async {
    try {
      await AppServices.instance.api.post(
        '/api/update-dietary-constraints',
        {'dietary_constraints': _constraintsController.text},
        auth: true,
      );

      if (!mounted) return;
      setState(() {
        // Update the first medical advice entry with new constraints
        if (_medicalAdvice.isNotEmpty) {
          _medicalAdvice[0]['dietary_constraints'] = _constraintsController.text;
        } else {
          _medicalAdvice.add({'dietary_constraints': _constraintsController.text});
        }
        _editingConstraints = false;
      });

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Dietary constraints updated successfully')),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving dietary constraints: $e')),
      );
    }
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
            _buildMedicalAdviceSection(context),
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
                  ).textTheme.bodyMedium?.copyWith(color: Colors.grey, fontSize: 16.8),
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
                              fontSize: 14.4,
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
                      ).textTheme.bodySmall?.copyWith(color: Colors.grey, fontSize: 14.4),
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
            Text(
              'Health Conditions',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            if (!_editingHealth) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.cardGreen,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(
                          Icons.medical_services,
                          color: AppColors.primary,
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Health Conditions',
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                              color: AppColors.primaryDark,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () {
                            setState(() {
                              _editingHealth = true;
                            });
                          },
                          icon: const Icon(Icons.edit, size: 18, color: AppColors.primary),
                          tooltip: 'Edit',
                          constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                          padding: EdgeInsets.zero,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
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
                              backgroundColor: Colors.white,
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
                  ],
                ),
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
              if (!_editingConstraints) ...[
                if (_medicalAdvice.any((a) =>
                    (a['dietary_constraints'] as String? ?? '').isNotEmpty)) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.cardGreen,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(
                              Icons.restaurant_outlined,
                              color: AppColors.primary,
                              size: 20,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Dietary Constraints',
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.primaryDark,
                                ),
                              ),
                            ),
                            IconButton(
                              onPressed: () {
                                setState(() {
                                  _editingConstraints = true;
                                });
                              },
                              icon: const Icon(Icons.edit, size: 18, color: AppColors.primary),
                              tooltip: 'Edit',
                              constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                              padding: EdgeInsets.zero,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        ..._medicalAdvice
                            .where((a) =>
                                (a['dietary_constraints'] as String? ?? '').isNotEmpty)
                            .map((a) => _buildConstraintsList(
                                  context,
                                  a['dietary_constraints'] as String,
                                )),
                      ],
                    ),
                  ),
                ] else ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.info_outline,
                          color: Colors.grey,
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'No dietary constraints saved in profile.',
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Colors.grey[600],
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () {
                            setState(() {
                              _editingConstraints = true;
                            });
                          },
                          icon: const Icon(Icons.edit, size: 18, color: Colors.grey),
                          tooltip: 'Edit',
                          constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                          padding: EdgeInsets.zero,
                        ),
                      ],
                    ),
                  ),
                ],
              ] else ...[
                const SizedBox(height: 12),
                TextField(
                  controller: _constraintsController,
                  maxLines: 3,
                  decoration: InputDecoration(
                    labelText: 'Dietary Constraints',
                    hintText: 'e.g. Low sodium, No sugar',
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
                      child: ElevatedButton.icon(
                        onPressed: _saveConstraintsChanges,
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
                        onPressed: () {
                          setState(() {
                            _initializeControllers();
                            _editingConstraints = false;
                          });
                        },
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
            ] else ...[
              TextField(
                controller: _healthController,
                maxLines: 3,
                decoration: InputDecoration(
                  labelText: 'Health Conditions (comma-separated)',
                  hintText: 'e.g. Diabetes, Hypertension, Celiac',
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
                      onPressed: _saveHealthChanges,
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
                      onPressed: () {
                        setState(() {
                          _healthController.text = _healthConditions.join(', ');
                          _editingHealth = false;
                        });
                      },
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

  Widget _buildDietarySection(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Dietary Profile',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            if (_preferences.isNotEmpty) ...[
              Text(
                'Preferences',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppColors.primaryDark,
                ),
              ),
              const SizedBox(height: 8),
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
                          Icons.favorite_outline,
                          size: 16,
                          color: AppColors.primary,
                        ),
                      ),
                    )
                    .toList(),
              ),
              const SizedBox(height: 12),
            ],
            if (_restrictions.isNotEmpty) ...[
              Text(
                'Restrictions',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppColors.primaryDark,
                ),
              ),
              const SizedBox(height: 8),
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
                        backgroundColor: Colors.red.shade100,
                        side: BorderSide.none,
                        avatar: const Icon(
                          Icons.cancel_outlined,
                          size: 16,
                          color: Colors.red,
                        ),
                      ),
                    )
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildMedicalAdviceSection(BuildContext context) {
    String joinField(String key) => _medicalAdvice
        .where((a) => (a[key] as String? ?? '').isNotEmpty)
        .map((a) => a[key] as String)
        .join('\n');

    final adviceText = joinField('medical_advice');
    final avoidText = joinField('avoid');
    final limitText = joinField('dietary_limit');
    final hasAdvice =
        adviceText.isNotEmpty || avoidText.isNotEmpty || limitText.isNotEmpty;

    Widget adviceRow(IconData icon, Color color, String label, String text) =>
        Container(
          margin: const EdgeInsets.only(bottom: 8),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AppColors.cardGreen,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, color: color, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: color,
                          ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      text,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppColors.textSecondary,
                          ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Medical Advice',
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            if (!hasAdvice)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.info_outline, color: Colors.grey, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'No medical advice saved in profile.',
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: Colors.grey[600]),
                      ),
                    ),
                  ],
                ),
              ),
            if (adviceText.isNotEmpty)
              adviceRow(
                Icons.local_hospital_outlined,
                AppColors.primary,
                'Advice',
                adviceText,
              ),
            if (avoidText.isNotEmpty)
              adviceRow(
                Icons.no_food_outlined,
                Colors.red.shade600,
                'Avoid',
                avoidText,
              ),
            if (limitText.isNotEmpty)
              adviceRow(
                Icons.monitor_heart_outlined,
                Colors.orange.shade700,
                'Dietary Limit',
                limitText,
              ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Dietary constraints helpers
  // ---------------------------------------------------------------------------

  static const _nutrientLabels = <String, String>{
    'sugar_g': 'Sugar',
    'sodium_mg': 'Sodium',
    'fiber_g': 'Fiber',
    'protein_g': 'Protein',
    'saturated_fat_g': 'Saturated Fat',
    'fat_g': 'Total Fat',
    'carbs_g': 'Carbohydrates',
    'calories': 'Calories',
    'cholesterol_mg': 'Cholesterol',
  };

  static const _nutrientUnits = <String, String>{
    'sugar_g': 'g',
    'sodium_mg': 'mg',
    'fiber_g': 'g',
    'protein_g': 'g',
    'saturated_fat_g': 'g',
    'fat_g': 'g',
    'carbs_g': 'g',
    'calories': 'kcal',
    'cholesterol_mg': 'mg',
  };

  static IconData _iconForNutrient(String key) {
    switch (key) {
      case 'sugar_g':
        return Icons.water_drop_outlined;
      case 'sodium_mg':
        return Icons.grain;
      case 'fiber_g':
        return Icons.eco_outlined;
      case 'protein_g':
        return Icons.fitness_center;
      case 'saturated_fat_g':
      case 'fat_g':
        return Icons.opacity;
      case 'carbs_g':
        return Icons.bakery_dining_outlined;
      case 'calories':
        return Icons.local_fire_department_outlined;
      default:
        return Icons.science_outlined;
    }
  }

  static String _fmtNum(num v) =>
      v == v.truncate() ? v.toInt().toString() : v.toStringAsFixed(1);

  Widget _limitChip(String label, Color fg, Color bg) => Container(
        margin: const EdgeInsets.only(left: 4),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label,
          style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: fg),
        ),
      );

  Widget _buildConstraintsList(BuildContext context, String rawJson) {
    Map<String, dynamic> parsed;
    try {
      parsed = jsonDecode(rawJson) as Map<String, dynamic>;
    } catch (_) {
      // Not valid JSON — show as plain text fallback
      return Text(
        rawJson,
        style: Theme.of(context)
            .textTheme
            .bodySmall
            ?.copyWith(color: AppColors.textSecondary),
      );
    }

    final rows = <Widget>[];
    for (final entry in parsed.entries) {
      final key = entry.key;
      final val = entry.value;
      if (val is! Map) continue;

      final label = _nutrientLabels[key] ?? key.replaceAll('_', ' ');
      final unit = _nutrientUnits[key] ?? '';
      final icon = _iconForNutrient(key);

      final chips = <Widget>[];
      if (val.containsKey('min')) {
        chips.add(_limitChip(
          '≥ ${_fmtNum(val['min'] as num)} $unit',
          Colors.teal.shade700,
          Colors.teal.shade50,
        ));
      }
      if (val.containsKey('max')) {
        chips.add(_limitChip(
          '≤ ${_fmtNum(val['max'] as num)} $unit',
          Colors.orange.shade800,
          Colors.orange.shade50,
        ));
      }

      if (chips.isEmpty) continue;

      rows.add(Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          children: [
            Icon(icon, size: 18, color: AppColors.primary),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                label,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.w500,
                      color: AppColors.textSecondary,
                    ),
              ),
            ),
            ...chips,
          ],
        ),
      ));
    }

    if (rows.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: rows,
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
}