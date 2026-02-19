import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../main.dart';
import '../../services/api_service.dart';
import '../../theme/app_theme.dart';

// ─── Tips shown in rotation ───────────────────────────────────────────────────
const _tips = [
  'Eating fiber-rich foods helps maintain stable blood sugar and keeps you full longer.',
  'Aim for at least 25–30 g of protein per meal to support muscle maintenance.',
  'Sodium below 2 300 mg/day helps keep blood pressure in a healthy range.',
  'Healthy fats from avocado, nuts, and olive oil support brain and heart health.',
  'Drinking water before meals can naturally reduce calorie intake.',
  'Colorful vegetables provide antioxidants that reduce inflammation.',
  'Regular meal timing helps regulate your hunger hormones.',
];

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool _loading = true;
  String? _error;
  String _username = '';
  Map<String, dynamic> _overview = {};
  Map<String, dynamic>? _nutritionAvg;
  List<Map<String, dynamic>> _recentRecipes = [];

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() { _loading = true; _error = null; });
    try {
      final storage = AppServices.instance.storage;
      final api = AppServices.instance.api;

      final name = await storage.getName();
      final username = await storage.getUsername();
      final data = await api.get('/dashboard') as Map<String, dynamic>;

      if (!mounted) return;
      setState(() {
        _username = name ?? username ?? 'there';
        _overview = data['overview'] as Map<String, dynamic>? ?? {};
        _nutritionAvg = data['nutrition_avg'] as Map<String, dynamic>?;
        _recentRecipes = List<Map<String, dynamic>>.from(
          data['recent_recipes'] as List? ?? [],
        );
        _loading = false;
      });
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() { _error = e.message; _loading = false; });
    } catch (_) {
      if (!mounted) return;
      setState(() { _error = 'Could not load data. Is the server running?'; _loading = false; });
    }
  }

  String _greeting() {
    final h = DateTime.now().hour;
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  }

  String get _tip => _tips[DateTime.now().weekday % _tips.length];

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
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
              onPressed: _loadData,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadData,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildGreetingCard(context),
            const SizedBox(height: 16),
            _buildStatsRow(context),
            const SizedBox(height: 16),
            if (_nutritionAvg != null) ...[
              _buildNutritionCard(context, _nutritionAvg!),
              const SizedBox(height: 16),
            ],
            if (_recentRecipes.isNotEmpty) ...[
              _buildRecentRecipes(context),
              const SizedBox(height: 16),
            ],
            _buildTipCard(context),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  // ─── Greeting ──────────────────────────────────────────────────────────────
  Widget _buildGreetingCard(BuildContext context) {
    final firstName = _username.split(' ').first;
    final saved = (_overview['saved_recipes'] as num?)?.toInt() ?? 0;
    final convs = (_overview['total_conversations'] as num?)?.toInt() ?? 0;
    return Card(
      color: AppColors.primary,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${_greeting()}, $firstName!',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: Colors.white, fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '$convs chats · $saved recipes saved',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white24, borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(Icons.eco, color: Colors.white, size: 32),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Stats row ─────────────────────────────────────────────────────────────
  Widget _buildStatsRow(BuildContext context) {
    final convs = (_overview['total_conversations'] as num?)?.toInt() ?? 0;
    final msgs = (_overview['total_messages'] as num?)?.toInt() ?? 0;
    final saved = (_overview['saved_recipes'] as num?)?.toInt() ?? 0;
    return Row(
      children: [
        Expanded(child: _StatCard(
          icon: Icons.chat_bubble_outline, label: 'Chats', value: '$convs',
        )),
        const SizedBox(width: 8),
        Expanded(child: _StatCard(
          icon: Icons.message_outlined, label: 'Messages', value: '$msgs',
        )),
        const SizedBox(width: 8),
        Expanded(child: _StatCard(
          icon: Icons.restaurant_menu, label: 'Saved', value: '$saved',
        )),
      ],
    );
  }

  // ─── Nutrition avg card ────────────────────────────────────────────────────
  Widget _buildNutritionCard(BuildContext context, Map<String, dynamic> avg) {
    final kcal = (avg['calories'] as num?)?.toDouble() ?? 0;
    final protein = (avg['protein_g'] as num?)?.toDouble() ?? 0;
    final carbs = (avg['carbs_g'] as num?)?.toDouble() ?? 0;
    final fat = (avg['fat_g'] as num?)?.toDouble() ?? 0;
    final fiber = (avg['fiber_g'] as num?)?.toDouble() ?? 0;
    final sodium = (avg['sodium_mg'] as num?)?.toDouble() ?? 0;
    final mealCount = (avg['meal_count'] as num?)?.toInt() ?? 0;

    final total = protein + carbs + fat;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  'Avg Nutrition per Saved Meal',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: AppColors.cardGreen,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '$mealCount meals',
                    style: const TextStyle(fontSize: 11, color: AppColors.primaryDark),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                // Macro pie chart
                SizedBox(
                  width: 140,
                  height: 140,
                  child: total > 0
                      ? PieChart(
                          PieChartData(
                            sectionsSpace: 2,
                            centerSpaceRadius: 38,
                            sections: [
                              PieChartSectionData(
                                value: protein,
                                color: AppColors.primary,
                                title: '',
                                radius: 28,
                              ),
                              PieChartSectionData(
                                value: carbs,
                                color: Colors.orange,
                                title: '',
                                radius: 28,
                              ),
                              PieChartSectionData(
                                value: fat,
                                color: Colors.blue.shade300,
                                title: '',
                                radius: 28,
                              ),
                            ],
                          ),
                        )
                      : const Center(child: Text('No data', style: TextStyle(color: Colors.grey))),
                ),
                const SizedBox(width: 16),
                // Calories + legend
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Calorie badge
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.local_fire_department, color: AppColors.primary, size: 16),
                            const SizedBox(width: 4),
                            Text(
                              '${kcal.toStringAsFixed(0)} kcal',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                color: AppColors.primaryDark,
                                fontSize: 15,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                      _MacroRow(color: AppColors.primary, label: 'Protein', grams: protein, total: total),
                      const SizedBox(height: 6),
                      _MacroRow(color: Colors.orange, label: 'Carbs', grams: carbs, total: total),
                      const SizedBox(height: 6),
                      _MacroRow(color: Colors.blue.shade300, label: 'Fat', grams: fat, total: total),
                      if (fiber > 0) ...[
                        const SizedBox(height: 6),
                        _MacroRow(color: Colors.green.shade300, label: 'Fiber', grams: fiber, total: total, showPct: false),
                      ],
                    ],
                  ),
                ),
              ],
            ),
            if (sodium > 0) ...[
              const SizedBox(height: 12),
              const Divider(height: 1),
              const SizedBox(height: 10),
              Row(
                children: [
                  const Icon(Icons.water_drop_outlined, size: 14, color: Colors.grey),
                  const SizedBox(width: 4),
                  Text(
                    'Avg sodium: ${sodium.toStringAsFixed(0)} mg/meal',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  const Spacer(),
                  if (sodium > 2300)
                    const Text(
                      '⚠ High',
                      style: TextStyle(fontSize: 11, color: Colors.orange, fontWeight: FontWeight.bold),
                    )
                  else
                    const Text(
                      '✓ OK',
                      style: TextStyle(fontSize: 11, color: AppColors.primary, fontWeight: FontWeight.bold),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ─── Recent saved recipes ──────────────────────────────────────────────────
  Widget _buildRecentRecipes(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Recently Saved Recipes',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            ..._recentRecipes.map((recipe) => _RecipeTile(recipe: recipe)),
          ],
        ),
      ),
    );
  }

  // ─── Tip card ──────────────────────────────────────────────────────────────
  Widget _buildTipCard(BuildContext context) {
    return Card(
      color: AppColors.cardGreen,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppColors.primary.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(Icons.lightbulb_outline, color: AppColors.primary, size: 28),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Daily Tip',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold, color: AppColors.primaryDark,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _tip,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppColors.textSecondary,
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
}

// ─── Macro legend row ─────────────────────────────────────────────────────────
class _MacroRow extends StatelessWidget {
  final Color color;
  final String label;
  final double grams;
  final double total;
  final bool showPct;

  const _MacroRow({
    required this.color,
    required this.label,
    required this.grams,
    required this.total,
    this.showPct = true,
  });

  @override
  Widget build(BuildContext context) {
    final pct = total > 0 ? (grams / total * 100).round() : 0;
    return Row(
      children: [
        Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(fontSize: 12)),
        const Spacer(),
        Text(
          '${grams.toStringAsFixed(1)}g${showPct ? '  $pct%' : ''}',
          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
        ),
      ],
    );
  }
}

// ─── Saved recipe tile ────────────────────────────────────────────────────────
class _RecipeTile extends StatelessWidget {
  final Map<String, dynamic> recipe;
  const _RecipeTile({required this.recipe});

  String _formatDate(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final now = DateTime.now();
      final diff = now.difference(dt);
      if (diff.inDays == 0) return 'Today';
      if (diff.inDays == 1) return 'Yesterday';
      if (diff.inDays < 7) return '${diff.inDays} days ago';
      return '${dt.day}/${dt.month}/${dt.year}';
    } catch (_) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    final name = recipe['recipe_name'] as String? ?? '';
    final savedAt = recipe['saved_at'] as String? ?? '';
    final kcal = (recipe['calories'] as num?)?.toDouble();
    final protein = (recipe['protein_g'] as num?)?.toDouble();
    final carbs = (recipe['carbs_g'] as num?)?.toDouble();
    final fat = (recipe['fat_g'] as num?)?.toDouble();
    final hasNutrition = kcal != null;

    return Padding(
      padding: const EdgeInsets.only(top: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(top: 2),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppColors.cardGreen, borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.restaurant_menu, color: AppColors.primary, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                ),
                const SizedBox(height: 4),
                if (hasNutrition)
                  Wrap(
                    spacing: 6,
                    children: [
                      _NutritionChip(label: '${kcal.toStringAsFixed(0)} kcal', color: AppColors.primary),
                      if (protein != null)
                        _NutritionChip(label: '${protein.toStringAsFixed(0)}g P', color: Colors.green.shade600),
                      if (carbs != null)
                        _NutritionChip(label: '${carbs.toStringAsFixed(0)}g C', color: Colors.orange),
                      if (fat != null)
                        _NutritionChip(label: '${fat.toStringAsFixed(0)}g F', color: Colors.blue.shade400),
                    ],
                  )
                else
                  const Text('No nutrition data', style: TextStyle(fontSize: 11, color: Colors.grey)),
                const SizedBox(height: 2),
                Text(
                  _formatDate(savedAt),
                  style: const TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _NutritionChip extends StatelessWidget {
  final String label;
  final Color color;
  const _NutritionChip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}

// ─── Stat card ────────────────────────────────────────────────────────────────
class _StatCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _StatCard({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            Icon(icon, color: AppColors.primary, size: 22),
            const SizedBox(height: 6),
            Text(
              value,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold, color: AppColors.primaryDark,
              ),
            ),
            Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey, fontSize: 11,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
