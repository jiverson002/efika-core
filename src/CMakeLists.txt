#-------------------------------------------------------------------------------
# SUBLIBRARY configuration
#-------------------------------------------------------------------------------
target_sources(${PROJECT_NAME}
  PRIVATE
    gc/cleanup.c
    gc/free.c
    gc/realloc.c
)
