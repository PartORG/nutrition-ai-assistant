// GENERATED FILE — DO NOT EDIT BY HAND.
//
// Produced from the FastAPI OpenAPI schema by scripts/gen_dart_models.py.
// Regenerate with:  uv run python scripts/gen_dart_models.py
//
// These exist so a field rename in src/adapters/rest/schemas.py becomes a Dart
// compile error instead of a runtime null on someone's phone. CI fails if this
// file drifts from the API.

class ConversationOut {
  final String conversationId;
  final String title;
  final String lastMessageAt;
  final String createdAt;

  const ConversationOut({
    required this.conversationId,
    required this.title,
    required this.lastMessageAt,
    required this.createdAt,
  });

  factory ConversationOut.fromJson(Map<String, dynamic> json) =>
      ConversationOut(
        conversationId: json['conversation_id'] as String,
        title: json['title'] as String,
        lastMessageAt: json['last_message_at'] as String,
        createdAt: json['created_at'] as String,
      );
}

/// Request body for updating dietary constraints.
class DietaryConstraintsUpdateRequest {
  final String dietaryConstraints;

  const DietaryConstraintsUpdateRequest({required this.dietaryConstraints});

  factory DietaryConstraintsUpdateRequest.fromJson(Map<String, dynamic> json) =>
      DietaryConstraintsUpdateRequest(
        dietaryConstraints: json['dietary_constraints'] as String,
      );
}

/// Request body for updating health conditions.
class HealthUpdateRequest {
  final String healthCondition;

  const HealthUpdateRequest({required this.healthCondition});

  factory HealthUpdateRequest.fromJson(Map<String, dynamic> json) =>
      HealthUpdateRequest(healthCondition: json['health_condition'] as String);
}

class ImageAnalysisOut {
  final List<String> detectedIngredients;
  final String? recommendationSummary;

  const ImageAnalysisOut({
    required this.detectedIngredients,
    this.recommendationSummary,
  });

  factory ImageAnalysisOut.fromJson(Map<String, dynamic> json) =>
      ImageAnalysisOut(
        detectedIngredients:
            (json['detected_ingredients'] as List<dynamic>? ?? const [])
                .map((e) => e as String)
                .toList(),
        recommendationSummary: json['recommendation_summary'] as String?,
      );
}

class LoginBody {
  final String login;
  final String password;

  const LoginBody({required this.login, required this.password});

  factory LoginBody.fromJson(Map<String, dynamic> json) => LoginBody(
    login: json['login'] as String,
    password: json['password'] as String,
  );
}

class MessageOut {
  final int? id;
  final String role;
  final String content;
  final String createdAt;

  const MessageOut({
    this.id,
    required this.role,
    required this.content,
    required this.createdAt,
  });

  factory MessageOut.fromJson(Map<String, dynamic> json) => MessageOut(
    id: json['id'] as int?,
    role: json['role'] as String,
    content: json['content'] as String,
    createdAt: json['created_at'] as String,
  );
}

/// Request body for updating food preferences.
class PreferencesUpdateRequest {
  final String preferences;

  const PreferencesUpdateRequest({required this.preferences});

  factory PreferencesUpdateRequest.fromJson(Map<String, dynamic> json) =>
      PreferencesUpdateRequest(preferences: json['preferences'] as String);
}

class RecommendationBody {
  final String query;

  const RecommendationBody({required this.query});

  factory RecommendationBody.fromJson(Map<String, dynamic> json) =>
      RecommendationBody(query: json['query'] as String);
}

class RecommendationOut {
  final String summary;
  final String rawRecommendations;

  const RecommendationOut({
    required this.summary,
    required this.rawRecommendations,
  });

  factory RecommendationOut.fromJson(Map<String, dynamic> json) =>
      RecommendationOut(
        summary: json['summary'] as String,
        rawRecommendations: json['raw_recommendations'] as String,
      );
}

class RefreshBody {
  final String token;

  const RefreshBody({required this.token});

  factory RefreshBody.fromJson(Map<String, dynamic> json) =>
      RefreshBody(token: json['token'] as String);
}

class RegisterBody {
  final String login;
  final String password;
  final String? name;
  final String? surname;
  final int? age;
  final String? gender;
  final String? caretaker;
  final String? healthCondition;

  const RegisterBody({
    required this.login,
    required this.password,
    this.name,
    this.surname,
    this.age,
    this.gender,
    this.caretaker,
    this.healthCondition,
  });

  factory RegisterBody.fromJson(Map<String, dynamic> json) => RegisterBody(
    login: json['login'] as String,
    password: json['password'] as String,
    name: json['name'] as String?,
    surname: json['surname'] as String?,
    age: json['age'] as int?,
    gender: json['gender'] as String?,
    caretaker: json['caretaker'] as String?,
    healthCondition: json['health_condition'] as String?,
  );
}

/// Request body for updating dietary restrictions / dislikes.
class RestrictionsUpdateRequest {
  final String restrictions;

  const RestrictionsUpdateRequest({required this.restrictions});

  factory RestrictionsUpdateRequest.fromJson(Map<String, dynamic> json) =>
      RestrictionsUpdateRequest(restrictions: json['restrictions'] as String);
}

class TokenResponse {
  final String accessToken;
  final String? tokenType;
  final int userId;
  final String role;

  const TokenResponse({
    required this.accessToken,
    this.tokenType,
    required this.userId,
    required this.role,
  });

  factory TokenResponse.fromJson(Map<String, dynamic> json) => TokenResponse(
    accessToken: json['access_token'] as String,
    tokenType: json['token_type'] as String?,
    userId: json['user_id'] as int,
    role: json['role'] as String,
  );
}

/// Request body for updating user profile.
class UserUpdateRequest {
  final String name;
  final int age;
  final String gender;
  final String caretaker;

  const UserUpdateRequest({
    required this.name,
    required this.age,
    required this.gender,
    required this.caretaker,
  });

  factory UserUpdateRequest.fromJson(Map<String, dynamic> json) =>
      UserUpdateRequest(
        name: json['name'] as String,
        age: json['age'] as int,
        gender: json['gender'] as String,
        caretaker: json['caretaker'] as String,
      );
}

class RatingUpdate {
  final int rating;

  const RatingUpdate({required this.rating});

  factory RatingUpdate.fromJson(Map<String, dynamic> json) =>
      RatingUpdate(rating: json['rating'] as int);
}
