# CPU unit tests: everything that doesn't need a GPU.
add_executable(unit_tests
  test_main.cpp
  test_basics.cpp
  test_attacks.cpp
  test_attack_variants.cpp
  test_fen.cpp
  test_board.cpp
  test_movegen.cpp
  test_eval.cpp
  test_eval_terms.cpp
  test_tune.cpp
  test_selfplay.cpp
  test_abdada.cpp
  test_move32.cpp
  test_nnue.cpp
  test_aspiration.cpp
  test_multipv.cpp
  test_pvs.cpp
  test_qchecks.cpp
  test_search.cpp
  test_parallel.cpp
  test_tt.cpp
  test_nmp.cpp
  test_lmr.cpp
  test_fut.cpp
  test_ext.cpp
  test_killers.cpp
  test_see.cpp)
target_link_libraries(unit_tests PRIVATE chessgpu_host chessgpu_warnings)
target_compile_definitions(unit_tests PRIVATE CHESSGPU_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME unit COMMAND unit_tests)
set_tests_properties(unit PROPERTIES LABELS unit)

# GPU tests: device code against the CPU, GPU perft against the suites.
add_executable(gpu_tests gpu_tests.cu gpu_search_tests.cu gpu_tt_tests.cu gpu_nmp_tests.cu gpu_lmr_tests.cu
  gpu_fut_tests.cu gpu_ext_tests.cu gpu_killers_tests.cu
  gpu_see_tests.cu gpu_abdada_tests.cu)
target_link_libraries(gpu_tests PRIVATE chessgpu_gpu chessgpu_warnings)
target_compile_definitions(gpu_tests PRIVATE CHESSGPU_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME gpu COMMAND gpu_tests)
set_tests_properties(gpu PROPERTIES LABELS gpu)

# Engine tests: UCI protocol, time management, backends (needs a GPU).
add_executable(engine_tests test_main.cpp engine_tests.cpp)
target_link_libraries(engine_tests PRIVATE chessgpu_engine chessgpu_warnings)
add_test(NAME engine COMMAND engine_tests)
set_tests_properties(engine PROPERTIES LABELS engine)

add_executable(smoke_test smoke_test.cu)
target_link_libraries(smoke_test PRIVATE chessgpu_core chessgpu_warnings)
add_test(NAME smoke COMMAND smoke_test)
set_tests_properties(smoke PROPERTIES LABELS gpu)

# Full perft suites through the engine's command line: every GPU
# board/model combination compiled in, plus the CPU (both boards, always).
# Optimized builds only.
set(suite_dir "${PROJECT_SOURCE_DIR}/tests/perft")
foreach(board IN LISTS CHESSGPU_BOARDS)
  foreach(model IN LISTS CHESSGPU_MODELS)
    add_test(NAME perft_${board}_${model} CONFIGURATIONS Release Profile
      COMMAND chessgpu perft-suite --board ${board} --model ${model} --max-nodes 10000000000)
    add_test(NAME perft_edge_${board}_${model} CONFIGURATIONS Release Profile
      COMMAND chessgpu perft-suite --file ${suite_dir}/edge_cases.epd --board ${board} --model ${model})
    set_tests_properties(perft_${board}_${model} perft_edge_${board}_${model} PROPERTIES LABELS perft)
  endforeach()
endforeach()
foreach(board std quad)
  add_test(NAME perft_${board}_cpu CONFIGURATIONS Release Profile
    COMMAND chessgpu perft-suite --board ${board} --model cpu --max-nodes 200000000)
  add_test(NAME perft_edge_${board}_cpu CONFIGURATIONS Release Profile
    COMMAND chessgpu perft-suite --file ${suite_dir}/edge_cases.epd --board ${board} --model cpu)
  set_tests_properties(perft_${board}_cpu perft_edge_${board}_cpu PROPERTIES LABELS perft)
endforeach()

# GPU tests again under compute-sanitizer: out-of-bounds and misaligned
# accesses (memcheck), shared-memory races (racecheck), invalid
# synchronization (synccheck) and reads of uninitialized device memory
# (initcheck). Run only these with: ctest -L sanitizer
find_program(COMPUTE_SANITIZER compute-sanitizer
  HINTS "${CUDAToolkit_BIN_DIR}/../compute-sanitizer" "${CUDAToolkit_BIN_DIR}")
if(COMPUTE_SANITIZER)
  add_test(NAME smoke_sanitizer
    COMMAND ${COMPUTE_SANITIZER} --error-exitcode 1 $<TARGET_FILE:smoke_test>)
  set_tests_properties(smoke_sanitizer PROPERTIES LABELS sanitizer)
  foreach(tool memcheck racecheck synccheck initcheck)
    add_test(NAME gpu_${tool}
      COMMAND ${COMPUTE_SANITIZER} --tool ${tool} --error-exitcode 1 $<TARGET_FILE:gpu_tests> --quick)
    set_tests_properties(gpu_${tool} PROPERTIES LABELS sanitizer)
  endforeach()
endif()

# The Python GUI's tests (gui/tests, pytest, headless), with the project venv
# (.venv; see gui/requirements.txt). The engine under test is this build's.
# Run only these with: ctest -L gui
find_program(CHESSGPU_GUI_PYTHON python
  HINTS "${CMAKE_SOURCE_DIR}/.venv/Scripts" "${CMAKE_SOURCE_DIR}/.venv/bin" NO_DEFAULT_PATH)
if(CHESSGPU_GUI_PYTHON)
  add_test(NAME gui COMMAND ${CHESSGPU_GUI_PYTHON} -m pytest -q -p no:cacheprovider
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/gui)
  set_tests_properties(gui PROPERTIES LABELS gui
    ENVIRONMENT "CHESSGPU_ENGINE=$<TARGET_FILE:chessgpu>;QT_QPA_PLATFORM=offscreen")
else()
  message(STATUS "GUI tests skipped: no Python in .venv (see gui/requirements.txt)")
endif()
