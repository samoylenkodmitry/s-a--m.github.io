---
layout: post
title: Android: ставим try-catch на все приложение
---
# Где в приложении спрятался метод main()?

Найти его можно опытным путем. Поставим брекпоинт в колбэк Application.onCreate и
обнаружим его в вершине стек-трейса, класс ActivityThread.
Вот она, "привычная" точка входа jvm-приложение:
```
    public static void main(String[] args) {
        ...
        Looper.prepareMainLooper();

        ActivityThread thread = new ActivityThread();
        thread.attach(false, startSeq);
        ...

        Looper.loop();

        throw new RuntimeException("Main thread loop unexpectedly exited");
    }

```
Пропустив нерелевантные нашей теме вещи, опишу что тут происходит:
1. `Looper.prepareMainLooper()` "подготавливается" Looper (создается thread-local экземпляр класса Looper)
2. `ActivityThread thread =...` Создается экземпляр ActivityThread
3. `thread.attach` Выполняется метод attach ActivityThread (создается экземпляр приложения и вызывается колбэк Application.onCreate)
4. `Looper.loop()` Запускается лупер. С этого момента лупер начинает опрашивать свою внутреннюю очередь и выполнять
из нее задания. Туда попадут всех остальные колбэки активити.
5. `throw new RuntimeException("Main thread loop unexpectedly exited")` в самой последней строке раскрывается
суть архитектуры сдк - на эту строчку приложение попасть не должно за всю свою жизнь. Если же мы в нее попали,
то приложение завершается ошибкой.
Таким образом, вырисовывается проблема и ее решение: если какой-нибудь произвольный колбэк Activity или, еще
интереснее, View, бросит ошибку, то все приложение упадет. Заранее подстраховаться от этого не получится никак.
Вообще в java есть универсальный глобальный механизм отлова ошибок:

```
		Thread.currentThread().setUncaughtExceptionHandler((t, e) -> ... ); //ловим все ошибки
```
Но представленная выше архитектура не позволит воспользоваться этим механизмом, т.к. сперва ошибка пробросится
в вызов Looper.loop(), завершив его, и лишь затем выйдет в main() и наш установленный хэндлер ошибок.
После того, как ошибка поймается хэндлером у нас остается проблема завершенного Looper.loop, и соответственно
приложения, больше _не_реагирующего_на_системные_колбэки_и_клики.

# Решение проблемы зависшего приложения
Чтобы активити снова реагировало на клики и колбэки достаточно опять запустить лупер:
```
		Looper.loop();
```

Итого, финальное решение. В Application.onCreate:
```
		Thread.currentThread().setUncaughtExceptionHandler((t, e) -> continueSafeLoop(e));
```

```
	public static void continueSafeLoop(final Throwable e) {
		Throwable error = e;
		while (true) {
			Assert.fail(error);
			try {
				final Looper looper = Looper.myLooper();
				new Handler(looper).removeCallbacksAndMessages(null);
				final MessageQueue queue = ReflectUtils.readField(looper, "mQueue");
				final Long ptr = ReflectUtils.readField(queue, "mPtr");
				final Boolean quitting = ReflectUtils.readField(queue, "mQuitting");
				if ((ptr == null || ptr != 0) && (quitting == null || !quitting.booleanValue())) {
					Looper.loop();
				} else {
					break; //тут просто детект завершения лупера; делается только рефлексивно
				}
			} catch (final Throwable err) {
				error = err;
			}
		}
	}
```
Теперь вы можете проэкспериментировать, бросив ошибку в любом из колбэков, и убедиться, что приложение не падает
и не зависает.
Конечно же, вам все равно придется абстрагироваться от прямых колбэков активити, т.к. в нем есть детект вызова super метода.
И обязательно нужно позаботиться об отсылке пойманных крашей в firebase/crashlytics, т.к. теперь
в стандартных отчетах об ошибках в консоли гугл плей не будет крашей.

P.S.: Примечательно, что класс ActivityThread не является потоком Thread и совсем не про андроидовский класс
Activity, а как раз про "активность" в смысле "набор действий", и создает не одно активити, а все приложение
Application. По сути он является делегатом всех системных колбэков приложения, отвечает за переходы его состояний.
Об этом также говорит его javadoc:
```
/**
 * This manages the execution of the main thread in an
 * application process, scheduling and executing activities,
 * broadcasts, and other operations on it as the activity
 * manager requests.
 *
 * {@hide}
 */
```
Неудачный выбор имени, как по мне :)
