import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:nutrition_ai_assistent/main.dart';

void main() {
  testWidgets('Login screen renders correctly', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});

    await tester.pumpWidget(const NutriAIApp(initialRoute: '/login'));
    await tester.pump();

    expect(find.text('NutriAI'), findsOneWidget);
    expect(find.text('Sign In'), findsOneWidget);
    expect(find.text("Don't have an account?"), findsOneWidget);
  });
}
