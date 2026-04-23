---
layout: leetcode-entry
title: "761. Special Binary String"
permalink: "/leetcode/problem/2026-02-20-761-special-binary-string/"
leetcode_ui: true
entry_slug: "2026-02-20-761-special-binary-string"
---

[761. Special Binary String](https://open.substack.com/pub/dmitriisamoilenko/p/20022026-761-special-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/20022026-761-special-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20022026-761-special-binary-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZNdaIujUycI)

![42c4c4c5-d78e-4f1e-a571-baa88a450791 (1).webp](/assets/leetcode_daily_images/4b61981e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1275

#### Problem TLDR

Swaps of 1M0-special substring to make string large #hard

#### Intuition

Didn't solve :)
```j
    // 30 minutes befor i go to hints
    // description unclear: prefix?
    // at least as many 1 as 0 what?
    // 11 ok
    // 01 not ok, prefix 0 has one zero and no ones
    // 1010 ok
    // 100 not ok, two zeros more than one
    // 110 ok
    // and the first condition: numbers are equal
    // 1010 ok
    // 1100 ok
    // 101 not ok
    // 110 not ok
    // 10 ok
    // 11101 not ok
    // 11101000 ok
    //
    // now the swap operation to make largest possible
    // meaning move ones to start
    // 11011000
    //  **
    //    ****
    //
    //
    // s length is only 50
    //
    // for every zero we try to:
    // 1. find first substring
    // 2. find next substring
    //
    // from zero we should only move left
    //
    // acceptance rate is too high 73% brainteaser?
    //
    // 8 minute
    //
    // 11011000
    //  **
    //    ****
    // let's try to go only 1 position to the left
    // that means next should always by 11
    // followed by 00
    // ok this is wrong
    // 1010101100
    // 1010110010 my
    // 1100101010 correct
    // 17 minute
    // 1010101100
    //     aabbbb
    // 1010110010 one change
    //            ok this can backtrack
    // how many times? maybe 50? what if we just repeat
    // another wrong answer
    // 110110100100
    // 110110100100 my
    // 111010010100 correct
    //              so the second chunk can be longer
    //              110100
    // 21 minute
    //              we should find the shortest chunk
    //              or should we try all of them?
    //              lets just take shortest
    // 30 minute wrong answer, go for hints
    //101110110011010000"
    //111100110100100010"
    //11101001100100010"
    // draw a line?? y coordinate?
```
Derived rules: starts with 1, ends with 0.
Slide and compute the balance. On each b==0 go deeper by a single char: "1(substring)0"

#### Approach

* why cut and then append 1..0? To go deeper.
* why can't cut 1..0 on the entire 's'? Because its not proven they are outer shell, they can be many chunks.

#### Complexity

- Time complexity:
$$O(n^2)$$, can be O(n)

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 37ms
    fun makeLargestSpecial(s: String): String = buildList {
        var b = 0; var j = 0
        for (i in s.indices) {
            b += 2 * (s[i]-'0') - 1
            if (b == 0) { add("1" + makeLargestSpecial(s.slice(j+1..<i)) + "0"); j = i + 1 }
        }
    }.sortedDescending().joinToString("")
```

