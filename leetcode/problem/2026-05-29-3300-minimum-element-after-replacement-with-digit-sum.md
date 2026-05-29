---
layout: leetcode-entry
title: "3300. Minimum Element After Replacement With Digit Sum"
permalink: "/leetcode/problem/2026-05-29-3300-minimum-element-after-replacement-with-digit-sum/"
leetcode_ui: true
entry_slug: "2026-05-29-3300-minimum-element-after-replacement-with-digit-sum"
---

[3300. Minimum Element After Replacement With Digit Sum](https://leetcode.com/problems/minimum-element-after-replacement-with-digit-sum/solutions/8300321/kotlin-rust-by-samoylenkodmitry-aflg/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29052026-3300-minimum-element-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FfOi6aizg2c)

https://dmitrysamoylenko.com/leetcode/

![29.05.2026.webp](/assets/leetcode_daily_images/29.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1374

#### Problem TLDR

Min of digits sum

#### Intuition

Compute the digits sum in a while loop, track the minimum.

#### Approach

* you can unroll the while loop

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minElement(n: IntArray) =
    n.minOf{"$it".sumOf{it-'0'}}
```
```rust
    pub fn min_element(n: Vec<i32>) -> i32 {
        n.iter().map(|x|x%10+x/10%10+x/100%10+x/1000%10+x/10000%10).min().unwrap()
    }
```

