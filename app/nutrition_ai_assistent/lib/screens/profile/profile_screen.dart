import 'dart:convert';

import 'package:flutter/material.dart';
import '../../main.dart';
import '../../services/api_service.dart';
import '../../theme/app_theme.dart';

class ProfileScreen extends StatefulWidget {
  final ValueNotifier<int>? refreshNotifier;

  const ProfileScreen({super.key, this.refreshNotifier});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _loading = true;
  String? _error;

  String _firstName = '';
  String _surname = '';
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
  bool _editingPreferences = false;
  bool _editingRestrictions = false;

  // Edit controllers — initialized at declaration so hot-reload never leaves them unset
  final _firstNameController    = TextEditingController();
  final _surnameController      = TextEditingController();
  final _ageController         = TextEditingController();
  final _genderController      = TextEditingController();
  final _caretakerController   = TextEditingController();
  final _healthController      = TextEditingController();
  final _constraintsController = TextEditingController();
  final _preferencesController = TextEditingController();
  final _restrictionsController = TextEditingController();

  @override
  void initState() {
    super.initState();
    widget.refreshNotifier?.addListener(_loadProfile);
    _loadProfile();
  }

  @override
  void dispose() {
    widget.refreshNotifier?.removeListener(_loadProfile);
    _firstNameController.dispose();
    _surnameController.dispose();
    _ageController.dispose();
    _genderController.dispose();
    _caretakerController.dispose();
    _healthController.dispose();
    _constraintsController.dispose();
    _preferencesController.dispose();
    _restrictionsController.dispose();
    super.dispose();
  }

  Future<void> _loadProfile() async {
    setState(() { _loading = true; _error = null; });
    try {
      final api = AppServices.instance.api;
      final profileData = await api.get('/api') as Map<String, dynamic>;

      if (!mounted) return;
      setState(() {
        final userMap = profileData['user'] as Map<String, dynamic>?;
        if (userMap != null) {
          _firstName = (userMap['name'] as String? ?? '').trim();
          _surname   = (userMap['surname'] as String? ?? '').trim();
          _username  = (userMap['user_name'] as String? ?? '').trim();
          _age       = userMap['age'] as int? ?? 0;
          _gender    = (userMap['gender'] as String? ?? '').trim();
          _caretaker = (userMap['caretaker'] as String? ?? '').trim();
        }

        final profiles = profileData['profiles'] as List? ?? [];
        if (profiles.isNotEmpty) {
          final latest = profiles.first as Map<String, dynamic>;
          _healthConditions = _split(latest['health_condition'] as String? ?? '');
          _preferences      = _split(latest['preferences']      as String? ?? '');
          _restrictions     = _split(latest['restrictions']     as String? ?? '');
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
      setState(() { _error = e.message; _loading = false; });
    } catch (_) {
      if (!mounted) return;
      setState(() { _error = 'Could not load profile. Is the server running?'; _loading = false; });
    }
  }

  void _initializeControllers() {
    _firstNameController.text  = _firstName;
    _surnameController.text     = _surname;
    _ageController.text       = _age > 0 ? _age.toString() : '';
    _genderController.text    = _gender;
    _caretakerController.text = _caretaker;
    _healthController.text    = _healthConditions.join(', ');
    _constraintsController.text = _medicalAdvice
        .where((a) => (a['dietary_constraints'] as String? ?? '').isNotEmpty)
        .map((a) => a['dietary_constraints'] as String)
        .join(', ');
    _preferencesController.text = _preferences.join(', ');
    _restrictionsController.text = _restrictions.join(', ');
  }

  Future<void> _saveHeaderChanges() async {
    try {
      await AppServices.instance.api.post('/api/update', {
        'name':     _firstNameController.text,
        'surname':  _surnameController.text,
        'age':      int.tryParse(_ageController.text) ?? 0,
        'gender':   _genderController.text,
        'caretaker': _caretakerController.text,
      }, auth: true);
      if (!mounted) return;
      setState(() {
        _firstName = _firstNameController.text;
        _surname   = _surnameController.text;
        _age         = int.tryParse(_ageController.text) ?? 0;
        _gender      = _genderController.text;
        _caretaker   = _caretakerController.text;
        _editingHeader = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Profile updated successfully')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  Future<void> _saveHealthChanges() async {
    try {
      await AppServices.instance.api.post(
        '/api/update-health',
        {'health_condition': _healthController.text},
        auth: true,
      );
      if (!mounted) return;
      // Reload the full profile so _medicalAdvice reflects the server state
      // (the backend clears the cached medical advice when health conditions
      // change so it can be regenerated on the next recommendation request).
      await _loadProfile();
      if (!mounted) return;
      setState(() => _editingHealth = false);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Health conditions updated. Medical advice will refresh on your next recommendation.')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
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
        if (_medicalAdvice.isNotEmpty) {
          _medicalAdvice[0]['dietary_constraints'] = _constraintsController.text;
        } else {
          _medicalAdvice.add({'dietary_constraints': _constraintsController.text});
        }
        _editingConstraints = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Dietary constraints updated successfully')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  Future<void> _savePreferencesChanges() async {
    try {
      await AppServices.instance.api.post(
        '/api/update-preferences',
        {'preferences': _preferencesController.text},
        auth: true,
      );
      if (!mounted) return;
      setState(() {
        _preferences = _preferencesController.text
            .split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
        _editingPreferences = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Preferences updated successfully')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  Future<void> _saveRestrictionsChanges() async {
    try {
      await AppServices.instance.api.post(
        '/api/update-restrictions',
        {'restrictions': _restrictionsController.text},
        auth: true,
      );
      if (!mounted) return;
      setState(() {
        _restrictions = _restrictionsController.text
            .split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
        _editingRestrictions = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Restrictions updated successfully')));
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.message}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  List<String> _split(String value) => value
      .split(',').map((s) => s.trim()).where((s) => s.isNotEmpty && s != 'None').toList();

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

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
            _buildHealthSection(context),
            const SizedBox(height: 16),
            _buildPreferencesCard(context),
            const SizedBox(height: 16),
            _buildRestrictionsCard(context),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Header card
  // ---------------------------------------------------------------------------

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
                    onPressed: () => setState(() => _editingHeader = true),
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
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(_firstName.isNotEmpty ? _firstName : (_username.isNotEmpty ? _username : 'User'),
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
                  if (_surname.isNotEmpty) ...[
                    const SizedBox(width: 8),
                    Text(_surname,
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
                  ],
                ],
              ),
              if (_username.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text('@$_username',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey, fontSize: 16.8)),
              ],
              if (demoBadges.isNotEmpty) ...[
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: demoBadges.map((label) => Chip(
                    label: Text(label, style: const TextStyle(fontSize: 14.4, color: AppColors.primaryDark)),
                    backgroundColor: AppColors.cardGreen,
                    side: BorderSide.none,
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  )).toList(),
                ),
              ],
              if (_caretaker.isNotEmpty) ...[
                const SizedBox(height: 10),
                const Divider(height: 1),
                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.people_outline, size: 16, color: Colors.grey),
                    const SizedBox(width: 6),
                    Text('Caretaker: $_caretaker',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey, fontSize: 14.4)),
                  ],
                ),
              ],
            ] else ...[
              Row(children: [
                Expanded(child: _buildTextField(_firstNameController, 'First Name')),
                const SizedBox(width: 12),
                Expanded(child: _buildTextField(_surnameController, 'Last Name')),
              ]),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(child: _buildTextField(_ageController, 'Age', keyboardType: TextInputType.number)),
                const SizedBox(width: 12),
                Expanded(child: _buildTextField(_genderController, 'Gender')),
              ]),
              const SizedBox(height: 12),
              _buildTextField(_caretakerController, 'Caretaker'),
              const SizedBox(height: 16),
              _buildSaveCancel(_saveHeaderChanges, () => setState(() {
                _editingHeader = false;
                _initializeControllers();
              })),
            ],
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Health Conditions + Medical Advice + Dietary Constraints (single card)
  // ---------------------------------------------------------------------------

  Widget _buildHealthSection(BuildContext context) {
    // Medical advice content
    String joinField(String key) => _medicalAdvice
        .where((a) => (a[key] as String? ?? '').isNotEmpty)
        .map((a) => a[key] as String)
        .join('\n');
    final adviceText = joinField('medical_advice');
    final avoidText  = joinField('avoid');
    final limitText  = joinField('dietary_limit');
    final hasAdvice  = adviceText.isNotEmpty || avoidText.isNotEmpty || limitText.isNotEmpty;
    final hasDietaryConstraints = _medicalAdvice.any(
        (a) => (a['dietary_constraints'] as String? ?? '').isNotEmpty);

    Widget adviceRow(IconData icon, Color color, String label, String text) => Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: AppColors.cardGreen, borderRadius: BorderRadius.circular(12)),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.bold, color: color)),
                const SizedBox(height: 2),
                Text(text, style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppColors.textSecondary)),
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
            Text('Health Conditions',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),

            // ── Health Conditions edit / view ───────────────────────────
            if (!_editingHealth) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(color: AppColors.cardGreen, borderRadius: BorderRadius.circular(12)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      const Icon(Icons.medical_services, color: AppColors.primary, size: 20),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text('Health Conditions',
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                fontWeight: FontWeight.w600, color: AppColors.primaryDark)),
                      ),
                      IconButton(
                        onPressed: () => setState(() => _editingHealth = true),
                        icon: const Icon(Icons.edit, size: 18, color: AppColors.primary),
                        tooltip: 'Edit',
                        constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                        padding: EdgeInsets.zero,
                      ),
                    ]),
                    const SizedBox(height: 8),
                    if (_healthConditions.isNotEmpty)
                      Wrap(
                        spacing: 8, runSpacing: 8,
                        children: _healthConditions.map((c) => Chip(
                          label: Text(c, style: const TextStyle(color: AppColors.primaryDark, fontSize: 13)),
                          backgroundColor: Colors.white,
                          side: BorderSide.none,
                          avatar: const Icon(Icons.medical_services, size: 16, color: AppColors.primary),
                        )).toList(),
                      )
                    else
                      Text('No health conditions set.',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(color: AppColors.cardGreen, borderRadius: BorderRadius.circular(12)),
                child: Row(children: [
                  const Icon(Icons.medical_services_outlined, color: AppColors.primary, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Your dietary recommendations are personalised based on your health conditions.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppColors.textSecondary),
                    ),
                  ),
                ]),
              ),
            ] else ...[
              _buildTextField(_healthController, 'Health Conditions (comma-separated)',
                  hint: 'e.g. Diabetes, Hypertension, Celiac', maxLines: 3),
              const SizedBox(height: 16),
              _buildSaveCancel(_saveHealthChanges, () => setState(() {
                _healthController.text = _healthConditions.join(', ');
                _editingHealth = false;
              })),
            ],

            // ── Medical Advice (moved here from separate card) ──────────
            if (!_editingHealth && hasAdvice) ...[
              const SizedBox(height: 12),
              Text('Medical Advice',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600, color: AppColors.primaryDark)),
              const SizedBox(height: 8),
              if (adviceText.isNotEmpty)
                adviceRow(Icons.local_hospital_outlined, AppColors.primary, 'Advice', adviceText),
              if (avoidText.isNotEmpty)
                adviceRow(Icons.no_food_outlined, Colors.red.shade600, 'Avoid', avoidText),
              if (limitText.isNotEmpty)
                adviceRow(Icons.monitor_heart_outlined, Colors.orange.shade700, 'Dietary Limit', limitText),
            ],

            // ── Dietary Constraints ─────────────────────────────────────
            if (!_editingHealth) ...[
              const SizedBox(height: 12),
              if (!_editingConstraints) ...[
                if (hasDietaryConstraints) ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(color: AppColors.cardGreen, borderRadius: BorderRadius.circular(12)),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(children: [
                          const Icon(Icons.restaurant_outlined, color: AppColors.primary, size: 20),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text('Dietary Constraints',
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    fontWeight: FontWeight.w600, color: AppColors.primaryDark)),
                          ),
                          IconButton(
                            onPressed: () => setState(() => _editingConstraints = true),
                            icon: const Icon(Icons.edit, size: 18, color: AppColors.primary),
                            tooltip: 'Edit',
                            constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                            padding: EdgeInsets.zero,
                          ),
                        ]),
                        const SizedBox(height: 8),
                        ..._medicalAdvice
                            .where((a) => (a['dietary_constraints'] as String? ?? '').isNotEmpty)
                            .map((a) => _buildConstraintsList(context, a['dietary_constraints'] as String)),
                      ],
                    ),
                  ),
                ] else ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(12)),
                    child: Row(children: [
                      const Icon(Icons.info_outline, color: Colors.grey, size: 20),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text('No dietary constraints saved.',
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey[600])),
                      ),
                      IconButton(
                        onPressed: () => setState(() => _editingConstraints = true),
                        icon: const Icon(Icons.edit, size: 18, color: Colors.grey),
                        tooltip: 'Edit',
                        constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                        padding: EdgeInsets.zero,
                      ),
                    ]),
                  ),
                ],
              ] else ...[
                _buildTextField(_constraintsController, 'Dietary Constraints',
                    hint: 'e.g. Low sodium, No sugar', maxLines: 3),
                const SizedBox(height: 12),
                _buildSaveCancel(_saveConstraintsChanges, () => setState(() {
                  _initializeControllers();
                  _editingConstraints = false;
                })),
              ],
            ],
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Preferences card
  // ---------------------------------------------------------------------------

  Widget _buildPreferencesCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              const Icon(Icons.favorite_outline, color: AppColors.primary, size: 22),
              const SizedBox(width: 8),
              Expanded(
                child: Text('My Preferences',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              ),
              if (!_editingPreferences)
                IconButton(
                  onPressed: () => setState(() => _editingPreferences = true),
                  icon: const Icon(Icons.edit, color: AppColors.primary, size: 20),
                  tooltip: 'Edit',
                  constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  padding: EdgeInsets.zero,
                ),
            ]),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: AppColors.cardGreen, borderRadius: BorderRadius.circular(10)),
              child: Row(children: [
                const Icon(Icons.info_outline, size: 16, color: AppColors.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'What do you love to eat? Add your favourite cuisines, ingredients, cooking styles, meal times, or flavour profiles. The AI uses this to suggest recipes you\'ll enjoy.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppColors.textSecondary),
                  ),
                ),
              ]),
            ),
            const SizedBox(height: 12),
            if (!_editingPreferences) ...[
              if (_preferences.isNotEmpty)
                Wrap(
                  spacing: 8, runSpacing: 8,
                  children: _preferences.map((p) => Chip(
                    label: Text(p, style: const TextStyle(color: AppColors.primaryDark, fontSize: 13)),
                    backgroundColor: AppColors.cardGreen,
                    side: BorderSide.none,
                    avatar: const Icon(Icons.favorite, size: 14, color: AppColors.primary),
                  )).toList(),
                )
              else
                Row(children: [
                  Icon(Icons.add_circle_outline, size: 18, color: Colors.grey.shade400),
                  const SizedBox(width: 6),
                  Text('No preferences set yet — tap edit to add some.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
                ]),
            ] else ...[
              _buildTextField(_preferencesController, 'Preferences (comma-separated)',
                  hint: 'e.g. Italian cuisine, chicken, quick meals, no onions, breakfast bowls',
                  maxLines: 3),
              const SizedBox(height: 12),
              _buildSaveCancel(_savePreferencesChanges, () => setState(() {
                _preferencesController.text = _preferences.join(', ');
                _editingPreferences = false;
              })),
            ],
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Restrictions card
  // ---------------------------------------------------------------------------

  Widget _buildRestrictionsCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              const Icon(Icons.block_outlined, color: Colors.red, size: 22),
              const SizedBox(width: 8),
              Expanded(
                child: Text('Restrictions & Dislikes',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              ),
              if (!_editingRestrictions)
                IconButton(
                  onPressed: () => setState(() => _editingRestrictions = true),
                  icon: const Icon(Icons.edit, color: AppColors.primary, size: 20),
                  tooltip: 'Edit',
                  constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  padding: EdgeInsets.zero,
                ),
            ]),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: Colors.red.shade50, borderRadius: BorderRadius.circular(10)),
              child: Row(children: [
                Icon(Icons.info_outline, size: 16, color: Colors.red.shade400),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'What should the AI avoid? Add foods you dislike, allergies, intolerances, or dietary restrictions (e.g. vegan, gluten-free, no shellfish).',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.red.shade700),
                  ),
                ),
              ]),
            ),
            const SizedBox(height: 12),
            if (!_editingRestrictions) ...[
              if (_restrictions.isNotEmpty)
                Wrap(
                  spacing: 8, runSpacing: 8,
                  children: _restrictions.map((r) => Chip(
                    label: Text(r, style: const TextStyle(color: Colors.red, fontSize: 13)),
                    backgroundColor: Colors.red.shade50,
                    side: BorderSide.none,
                    avatar: Icon(Icons.cancel_outlined, size: 14, color: Colors.red.shade400),
                  )).toList(),
                )
              else
                Row(children: [
                  Icon(Icons.add_circle_outline, size: 18, color: Colors.grey.shade400),
                  const SizedBox(width: 6),
                  Text('No restrictions set yet — tap edit to add some.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
                ]),
            ] else ...[
              _buildTextField(_restrictionsController, 'Restrictions (comma-separated)',
                  hint: 'e.g. no zucchini, vegan, gluten-free, shellfish allergy, low FODMAP',
                  maxLines: 3),
              const SizedBox(height: 12),
              _buildSaveCancel(_saveRestrictionsChanges, () => setState(() {
                _restrictionsController.text = _restrictions.join(', ');
                _editingRestrictions = false;
              })),
            ],
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Shared helpers
  // ---------------------------------------------------------------------------

  Widget _buildTextField(
    TextEditingController controller,
    String label, {
    String? hint,
    int maxLines = 1,
    TextInputType keyboardType = TextInputType.text,
  }) =>
      TextField(
        controller: controller,
        maxLines: maxLines,
        keyboardType: keyboardType,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        ),
      );

  Widget _buildSaveCancel(VoidCallback onSave, VoidCallback onCancel) => Row(
    children: [
      Expanded(
        child: ElevatedButton.icon(
          onPressed: onSave,
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
          onPressed: onCancel,
          icon: const Icon(Icons.close),
          label: const Text('Cancel'),
          style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12)),
        ),
      ),
    ],
  );

  // ---------------------------------------------------------------------------
  // Dietary constraints JSON renderer
  // ---------------------------------------------------------------------------

  static const _nutrientLabels = <String, String>{
    'sugar_g': 'Sugar',         'sodium_mg': 'Sodium',
    'fiber_g': 'Fiber',         'protein_g': 'Protein',
    'saturated_fat_g': 'Sat. Fat', 'fat_g': 'Total Fat',
    'carbs_g': 'Carbs',         'calories': 'Calories',
    'cholesterol_mg': 'Cholesterol',
  };

  static const _nutrientUnits = <String, String>{
    'sugar_g': 'g',     'sodium_mg': 'mg',  'fiber_g': 'g',
    'protein_g': 'g',   'saturated_fat_g': 'g', 'fat_g': 'g',
    'carbs_g': 'g',     'calories': 'kcal', 'cholesterol_mg': 'mg',
  };

  static IconData _iconForNutrient(String key) {
    switch (key) {
      case 'sugar_g':        return Icons.water_drop_outlined;
      case 'sodium_mg':      return Icons.grain;
      case 'fiber_g':        return Icons.eco_outlined;
      case 'protein_g':      return Icons.fitness_center;
      case 'saturated_fat_g':
      case 'fat_g':          return Icons.opacity;
      case 'carbs_g':        return Icons.bakery_dining_outlined;
      case 'calories':       return Icons.local_fire_department_outlined;
      default:               return Icons.science_outlined;
    }
  }

  static String _fmtNum(num v) =>
      v == v.truncate() ? v.toInt().toString() : v.toStringAsFixed(1);

  Widget _limitChip(String label, Color fg, Color bg) => Container(
    margin: const EdgeInsets.only(left: 4),
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
    child: Text(label, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: fg)),
  );

  Widget _buildConstraintsList(BuildContext context, String rawJson) {
    Map<String, dynamic> parsed;
    try {
      parsed = jsonDecode(rawJson) as Map<String, dynamic>;
    } catch (_) {
      return Text(rawJson,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppColors.textSecondary));
    }

    final rows = <Widget>[];
    for (final entry in parsed.entries) {
      final key = entry.key;
      final val = entry.value;
      if (val is! Map) continue;

      final label = _nutrientLabels[key] ?? key.replaceAll('_', ' ');
      final unit  = _nutrientUnits[key] ?? '';
      final icon  = _iconForNutrient(key);

      final chips = <Widget>[];
      if (val.containsKey('min')) {
        chips.add(_limitChip('≥ ${_fmtNum(val['min'] as num)} $unit',
            Colors.teal.shade700, Colors.teal.shade50));
      }
      if (val.containsKey('max')) {
        chips.add(_limitChip('≤ ${_fmtNum(val['max'] as num)} $unit',
            Colors.orange.shade800, Colors.orange.shade50));
      }
      if (chips.isEmpty) continue;

      rows.add(Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          children: [
            Icon(icon, size: 18, color: AppColors.primary),
            const SizedBox(width: 8),
            Expanded(
              child: Text(label,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.w500, color: AppColors.textSecondary)),
            ),
            ...chips,
          ],
        ),
      ));
    }

    if (rows.isEmpty) return const SizedBox.shrink();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: rows);
  }
}
