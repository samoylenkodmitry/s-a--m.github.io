---
layout: leetcode-entry
title: "1872. Stone Game VIII"
permalink: "/leetcode/problem/2026-08-24-1872-stone-game-viii/"
leetcode_ui: true
entry_slug: "2026-08-24-1872-stone-game-viii"
---

[1872. Stone Game VIII](https://leetcode.com/problems/stone-game-viii/solutions/8479603/kotlin-rust-by-samoylenkodmitry-0i0m/) hard
[substack](https://dmitriisamoilenko.substack.com/p/24082026-1872-stone-game-viii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cyIXUg-BSt4)

https://dmitrysamoylenko.com/leetcode/

![24.08.2026.webp](/assets/leetcode_daily_images/24.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1461

#### Problem TLDR

Alice maximize and Bob minimize the (A-B)

#### Intuition

Didn't solve because misunderstood the problem: *it is not the abs difference*. The (A-B) is symmetrical: Alice wants minimize Bob, Bob wants minimize Alice, both want maximize themselves. After that, problem is a trivial take/skip dp.

#### Approach

* final condition must not be zero, because the last value could be negative

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun stoneGameVIII(s: IntArray) = s.scan(0, Int::plus).run {
        (size-2 downTo 2).fold(last()) { ans, i -> max(ans, get(i)-ans) }
    }
```
```rust
    pub fn stone_game_viii(mut s: Vec<i32>) -> i32 {
        for i in 1..s.len() { s[i] += s[i - 1] }
        let last = s.pop().unwrap();
        s[1..].iter().rfold(last, |ans, &x| ans.max(x - ans))
    }
```

