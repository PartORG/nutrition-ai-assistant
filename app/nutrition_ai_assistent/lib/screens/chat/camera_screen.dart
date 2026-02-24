import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Full-screen camera preview for taking a food photo.
///
/// Returns the captured [XFile] via [Navigator.pop] when the shutter is
/// pressed, or `null` when the user cancels with the back button.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;
  bool _initializing = true;
  bool _capturing = false;
  String? _error;

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCameras();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _startController(_cameras[_cameraIndex]);
    }
  }

  // ── Camera setup ───────────────────────────────────────────────────────────

  Future<void> _initCameras() async {
    try {
      _cameras = await availableCameras();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Could not enumerate cameras: $e';
        _initializing = false;
      });
      return;
    }

    if (_cameras.isEmpty) {
      if (!mounted) return;
      setState(() {
        _error = 'No cameras found on this device.';
        _initializing = false;
      });
      return;
    }

    await _startController(_cameras[_cameraIndex]);
  }

  Future<void> _startController(CameraDescription camera) async {
    final ctrl = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    try {
      await ctrl.initialize();
    } catch (e) {
      ctrl.dispose();
      if (!mounted) return;
      setState(() {
        _error = 'Camera initialisation failed: $e';
        _initializing = false;
      });
      return;
    }
    if (!mounted) {
      ctrl.dispose();
      return;
    }
    _controller?.dispose();
    setState(() {
      _controller = ctrl;
      _initializing = false;
      _error = null;
    });
  }

  // ── Actions ────────────────────────────────────────────────────────────────

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _capturing || _initializing) return;
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    setState(() => _initializing = true);
    await _startController(_cameras[_cameraIndex]);
  }

  Future<void> _capture() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized || _capturing) return;
    setState(() => _capturing = true);
    try {
      final file = await ctrl.takePicture();
      if (!mounted) return;
      Navigator.pop(context, file);
    } catch (e) {
      if (!mounted) return;
      setState(() => _capturing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not capture photo: $e')),
      );
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Take Photo'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.no_photography_outlined,
                color: Colors.white60,
                size: 64,
              ),
              const SizedBox(height: 16),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 15),
              ),
            ],
          ),
        ),
      );
    }

    if (_initializing ||
        _controller == null ||
        !_controller!.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Live preview ────────────────────────────────────────────────────
        CameraPreview(_controller!),

        // ── Shutter button ──────────────────────────────────────────────────
        Positioned(
          bottom: 40,
          left: 0,
          right: 0,
          child: Center(
            child: GestureDetector(
              onTap: _capturing ? null : _capture,
              child: Container(
                width: 76,
                height: 76,
                decoration: BoxDecoration(
                  color: _capturing ? Colors.grey.shade400 : Colors.white,
                  shape: BoxShape.circle,
                  border: Border.all(color: AppColors.primary, width: 4),
                ),
                child: _capturing
                    ? const Padding(
                        padding: EdgeInsets.all(20),
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                          color: AppColors.primary,
                        ),
                      )
                    : const Icon(
                        Icons.camera_alt,
                        color: AppColors.primary,
                        size: 36,
                      ),
              ),
            ),
          ),
        ),

        // ── Switch-camera button (bottom-right) ─────────────────────────────
        if (_cameras.length > 1)
          Positioned(
            bottom: 52,
            right: 32,
            child: GestureDetector(
              onTap: (_initializing || _capturing) ? null : _switchCamera,
              child: Container(
                width: 48,
                height: 48,
                decoration: const BoxDecoration(
                  color: Colors.black45,
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.flip_camera_ios,
                  color: Colors.white,
                  size: 26,
                ),
              ),
            ),
          ),
      ],
    );
  }
}
