---
layout: post
title: Don't Lie to Me
---
# Do You Like When Your Code Lies to You?

It turns out that any frontend, as it becomes more complex with tasks and frameworks, tends to devolve into an incomprehensible something. 
Take, for example, my favorite case with JavaScript:
```
Console.log("3"+2); //5
Console.log("3"-2); //1
```
and transfer it to Android with the sweet, sweet Kotlin:

```
operator fun String.minus(n : Int) = this.toInt() - n
    
fun main() {
   println("3" + 2) //5
   println("3" - 2) //1
}
```
When was the last time you saw something like this in strict but fair mother Java? (actually, there are plenty of traps in it, we just learned them)

And today, I was delighted with another quiz. Guess what the following line in XML does:

```
app:myCustomBinding="@{SystemUiHider.SInsetsHolder.top + ResourceUtils.dipToPx(context, @dimen/statement_padding_16dp)}"
```
If your answer is: it adds together two numbers `SystemUiHider.SInsetsHolder.top` and `ResourceUtils.dipToPx(context, @dimen/statement_padding_16dp)` and 
passes them to a custom binding adapter, then congratulations. You are behind the times, okay boomer, go code in your Delphi and don't bother the cool kids.

The answer from gen-z: since the variable `SInsetsHolder` is of type `androidx.databinding.BaseObservable`, the "smart" code generation will subscribe the entire expression
in the brackets `@{..}` to the change of the `top` parameter and will call it when necessary. Perhaps the main mistake of the data binding designers
was that they tried to imitate the syntax of simple Java in XML. When in fact, their own DSL syntax was needed. Then there would be no misunderstandings.
  
  
Conclusion:
Of course, sugar can speed up the coding process and make it `fun`. But the price is a complete loss of understanding of what is actually happening in the application.





