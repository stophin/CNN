@echo off

@set FILE=%*
@set FILE=%FILE:/=\%

rem echo %FILE%
del %FILE%