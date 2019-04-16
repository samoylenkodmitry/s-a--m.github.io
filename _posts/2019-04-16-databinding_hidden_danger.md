---
layout: post
title: Databinding Episode I; Hidden Danger
---
# Databinding Эпизод I; Скрытая угроза

Иногда в повседневной разработке под Android сталкиваешься со странными, если не сказать мистическими багами. (стандартное вступление)

Наш QA заметил странный баг, который трое из наших разработчиков не могли заметить глядя на экран в упор. 
При переходе на экран очень быстро "промаргивала" часть экрана, которая затем скрывалась. Очень хороший FPS у тестировщика :)
Отловив в предложенном видео кадр с "промаргиванием" я усомнился в своей способности видеть этот мир, по крайней мере в сравнении 
с некоторыми людьми. Что ж, нужно править.

Забравшись в верстку обнаруживаю в общем-то безобидный кусок xml-ля:

```
				...
				<ru.ivi.uikit.UiKitGridLayout
				...
					android:visibility="@{authState.isUserAuthorized ? View.GONE : View.VISIBLE, default=gone}"
				...
```

Эта часть лэйаута должна была быть скрыта по умолчанию, а затем либо появляться, либо нет, в зависимости от статуса авторизации.
Но почему-то по умолчанию она показывалась и затем скрывалась как и должна для авторизованного юзера.

Похоже где-то в этой строке есть ошибка. Может быть ", default=gone"? Тут нам не подскажет документация, придется лезть в устройство
дата-байндинга:

```
       UserAuthorizedState authState = mAuthState;
        int authStateIsUserAuthorizedViewGONEViewVISIBLE = 0;
        boolean authStateIsUserAuthorized = false;

        if ((dirtyFlags & 0x3L) != 0) {
                if (authState != null) {
                    // read authState.isUserAuthorized
                    authStateIsUserAuthorized = authState.isUserAuthorized;
                }
            if((dirtyFlags & 0x3L) != 0) {
                if(authStateIsUserAuthorized) {
                        dirtyFlags |= 0x8L;
                }
                else {
                        dirtyFlags |= 0x4L;
                }
            }
                // read authState.isUserAuthorized ? View.GONE : View.VISIBLE
                authStateIsUserAuthorizedViewGONEViewVISIBLE = ((authStateIsUserAuthorized) ? (android.view.View.GONE) : (android.view.View.VISIBLE));
        }
        // batch finished
        if ((dirtyFlags & 0x3L) != 0) {
            // api target 1

            this.motivationToRegistration.setVisibility(authStateIsUserAuthorizedViewGONEViewVISIBLE);
        }
```
Если проследить за логикой этого сгенерированного кода, то видим, что authStateIsUserAuthorized по умолчанию false.
Если байндинг еще не произошел, то есть authState==null, то значение переменной не меняется. 
В результате переменная authStateIsUserAuthorizedViewGONEViewVISIBLE содержит View.VISIBLE.
А как же то что мы указали в коде ", default=gone"? 

Нет никакого default! Вот документация и там его нет https://developer.android.com/topic/libraries/data-binding/expressions смотрите сами!
Он есть только для строк и только в одном из ответом на stackoverflow https://stackoverflow.com/questions/39241191/error-with-default-value-in-databinding?rq=1

Такие дела. 

Теперь уберем из xml слово default и сравним сгенерированный код - ничего не поменялось.
Давайте просто сделаем вид, что default это наша фантазия о том как должен выглядеть databinding. 
А пока что заменим код в xml на что-то более надежное:

```
android:visibility="@{authState==null||authState.isUserAuthorized ? View.GONE : View.VISIBLE}"
```

Теперь никаким супер-зрением наш лэйаут не увидеть.


Вывод такой:
будьте осторожны с кодом внутри xml и databinding и еще осторожнее с Android SDK ༼ʘ̚ل͜ʘ̚༽
