include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)

add_executable(test_vectorizer test_vectorizer.cpp)
target_link_libraries(test_vectorizer PRIVATE neroued::vectorizer nanosvg GTest::gtest_main)
target_include_directories(test_vectorizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME test_vectorizer COMMAND test_vectorizer)

add_executable(test_vectorizer_potrace test_vectorizer_potrace.cpp)
target_link_libraries(test_vectorizer_potrace PRIVATE neroued::vectorizer nanosvg GTest::gtest_main)
target_include_directories(test_vectorizer_potrace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME test_vectorizer_potrace COMMAND test_vectorizer_potrace)

if(NV_BUILD_EVAL)
    add_executable(test_vectorize_eval ../eval/tests/test_vectorize_eval.cpp)
    target_link_libraries(test_vectorize_eval PRIVATE neroued::vectorizer_eval GTest::gtest_main)
    target_include_directories(test_vectorize_eval PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval/src)
    add_test(NAME test_vectorize_eval COMMAND test_vectorize_eval)
endif()
