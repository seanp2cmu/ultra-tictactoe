# UI 클래스 인스턴스 변수 목록

이 문서는 `UI` 클래스의 모든 인스턴스 변수를 정리하여 존재하지 않는 변수 참조 오류를 방지합니다.

## ⚠️ 중요: 새 변수 추가 시
새로운 인스턴스 변수를 추가할 때는 **반드시** `__init__` 메서드에서 초기화하세요!

```python
def __init__(self):
    # ...
    self.new_variable = initial_value  # ✅ 올바른 방법
```

메서드 내에서 갑자기 `self.new_variable`을 사용하면 AttributeError가 발생할 수 있습니다.

---

## Window & Display
- `WINDOW_WIDTH: int` - 창 너비 (1400)
- `WINDOW_HEIGHT: int` - 창 높이 (1000)
- `screen: pygame.Surface` - Pygame 화면 객체
- `clock: pygame.time.Clock` - FPS 제어용 시계

## Fonts
- `font_large: pygame.font.Font` - 큰 폰트 (48)
- `font_medium: pygame.font.Font` - 중간 폰트 (36)
- `font_small: pygame.font.Font` - 작은 폰트 (24)
- `font_tiny: pygame.font.Font` - 아주 작은 폰트 (18)

## Game State
- `mode: GameMode` - 현재 게임 모드
- `board: Optional[Board]` - 게임 보드
- `running: bool` - 게임 실행 중 여부
- `game_over: bool` - 게임 종료 여부
- `move_history: List[Tuple[int, int]]` - 수 기록
- `mode_name: str` - 모드 이름 문자열

## AI State
- `ai_agent: Optional[AlphaZeroAgent]` - AI 에이전트
- `ai_network: Optional[AlphaZeroNet]` - AI 네트워크
- `ai_player: int` - AI 플레이어 번호 (1 또는 2)
- `num_simulations: int` - MCTS 시뮬레이션 횟수
- `available_models: List[str]` - 사용 가능한 모델 경로 목록
- `selected_model_idx: int` - 선택된 모델 인덱스
- `player_goes_first: bool` - 플레이어 선공 여부

## Analysis State
- `show_analysis: bool` - 분석 표시 여부
- `top_n_moves: int` - 상위 N개 수 표시
- `analysis_data: Optional[Any]` - 분석 데이터

## Menu/UI State
- `selecting_model: bool` - 모델 선택 중 여부
- `selected_mode_for_model: Optional[GameMode]` - 선택한 모드
- `menu_buttons: List[Tuple[pygame.Rect, str]]` - 메뉴 버튼 목록
- `model_buttons: List[Tuple[pygame.Rect, Any]]` - 모델 버튼 목록
- `slider_rect: Optional[pygame.Rect]` - 슬라이더 영역
- `slider_handle_rect: Optional[pygame.Rect]` - 슬라이더 핸들
- `dragging_slider: bool` - 슬라이더 드래그 중
- `dragging_compare_games: bool` - 비교 게임 수 슬라이더 드래그 중
- `dragging_compare_sims: bool` - 비교 시뮬레이션 수 슬라이더 드래그 중
- `dragging_compare_temp: bool` - 비교 temperature 슬라이더 드래그 중
- `loading_game: bool` - 게임 로딩 중
- `saved_game_buttons: List[Tuple[pygame.Rect, str]]` - 저장된 게임 버튼
- `saved_games_list: List[str]` - 저장된 게임 목록

## Game History & Review
- `game_history: Optional[GameHistory]` - 게임 히스토리
- `move_list_panel: MoveListPanel` - 수 목록 패널
- `review_controls: ReviewControls` - 리뷰 컨트롤
- `review_model1_name: Optional[str]` - 리뷰 모드 Model 1 이름
- `review_model2_name: Optional[str]` - 리뷰 모드 Model 2 이름
- `review_first_player: Optional[str]` - 리뷰 모드 선공
- `from_comparison: bool` - 비교 결과에서 리뷰 진입 여부
- `back_button_rect: Optional[pygame.Rect]` - Back 버튼 영역

## Compare Models State
- `comparing_models: bool` - 모델 비교 중
- `compare_model1_idx: int` - 비교 Model 1 인덱스
- `compare_model2_idx: int` - 비교 Model 2 인덱스
- `compare_num_games: int` - 비교 게임 수
- `compare_simulations: int` - 비교 MCTS 시뮬레이션 수
- `compare_temperature: float` - 비교 시 temperature 값
- `model1_scroll_offset: int` - Model 1 리스트 스크롤 오프셋
- `model2_scroll_offset: int` - Model 2 리스트 스크롤 오프셋
- `game_details: List[Dict[str, Any]]` - 게임 상세 정보 목록
- `viewing_game_detail: Optional[Dict]` - 현재 보는 게임 상세
- `simulation_progress: int` - 시뮬레이션 진행도 (0-100)
- `current_game_num: int` - 현재 게임 번호
- `simulation_running: bool` - 시뮬레이션 실행 중
- `simulation_cancelled: bool` - 시뮬레이션 취소됨
- `compare_results: Optional[Dict[str, int]]` - 비교 결과

## Renderers & Managers
- `game_renderer: GameRenderer` - 게임 렌더러
- `menu_renderer: MenuRenderer` - 메뉴 렌더러
- `compare_renderer: CompareRenderer` - 비교 렌더러
- `compare_manager: CompareManager` - 비교 매니저 (시뮬레이션 로직)
- `turn_order_buttons: Optional[List[Tuple[pygame.Rect, str]]]` - 턴 순서 버튼 (동적 생성)

---

## 🛡️ 변수 사용 시 주의사항

1. **Optional 타입 변수**: `None` 체크 필수
   ```python
   if self.ai_agent is not None:
       self.ai_agent.select_action(...)
   ```

2. **동적 생성 변수**: 일부 버튼은 렌더링 시 생성됨
   ```python
   if hasattr(self, 'back_button_rect'):
       if self.back_button_rect.collidepoint(pos):
           ...
   ```

3. **새 변수 추가 시**: 반드시 `__init__`에 추가하고 이 문서도 업데이트!
