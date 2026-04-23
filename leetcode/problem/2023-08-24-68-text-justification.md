---
layout: leetcode-entry
title: "68. Text Justification"
permalink: "/leetcode/problem/2023-08-24-68-text-justification/"
leetcode_ui: true
entry_slug: "2023-08-24-68-text-justification"
---

[68. Text Justification](https://leetcode.com/problems/text-justification/description/) hard
[blog post](https://leetcode.com/problems/text-justification/solutions/3952534/kotlin-not-hard-just-corner-cases/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24082023-68-text-justification?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/a2960305.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/318

#### Problem TLDR

Spread `words` to lines, evenly spacing left->right, and left-spacing the last line

#### Intuition

Scan word by word, checking `maxWidth` overflow.

#### Approach

Separate word letters count and count of spaces.
To spread spaces left-evenly, iteratively add spaces one-by-one until `maxWidth` reached.
Using Kotlin built-in functions helps to reduce boilerplate:
* buildList
* buildString
* padEnd

#### Complexity

- Time complexity:
$$O(wn)$$

- Space complexity:
$$O(wn)$$

#### Code

```kotlin

    fun fullJustify(words: Array<String>, maxWidth: Int) = buildList<String> {
      val line = mutableListOf<String>()
      fun justifyLeft() = line.joinToString(" ").padEnd(maxWidth, ' ')
      var wLen = 0
      fun justifyFull() = buildString {
        val sp = IntArray(line.size - 1) { 1 }
        var i = 0
        var len = wLen + line.size - 1
        while (len++ < maxWidth && line.size > 1) sp[i++ % sp.size]++
        line.forEachIndexed { i, w ->
          append(w)
          if (i < sp.size) append(" ".repeat(sp[i]))
        }
      }
      words.forEachIndexed { i, w ->
        if (wLen + line.size + w.length > maxWidth) {
          add(if (line.size > 1) justifyFull() else justifyLeft())

          line.clear()
          wLen = 0
        }
        line += w
        wLen += w.length
      }
      add(justifyLeft())
    }

```

