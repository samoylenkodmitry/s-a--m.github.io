---
layout: leetcode-entry
title: "2996. Smallest Missing Integer Greater Than Sequential Prefix Sum"
permalink: "/leetcode/problem/2026-08-11-2996-smallest-missing-integer-greater-than-sequential-prefix-sum/"
leetcode_ui: true
entry_slug: "2026-08-11-2996-smallest-missing-integer-greater-than-sequential-prefix-sum"
---

[2996. Smallest Missing Integer Greater Than Sequential Prefix Sum](https://leetcode.com/problems/smallest-missing-integer-greater-than-sequential-prefix-sum/solutions/8454131/kotlin-rust-by-samoylenkodmitry-d7w3/) easy
[substack](https://dmitriisamoilenko.substack.com/p/11082026-2996-smallest-missing-integer?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/k6CUL8JKEBg)

https://dmitrysamoylenko.com/leetcode/

![11.08.2026.webp](/assets/leetcode_daily_images/11.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1448

#### Problem TLDR

First prefix sum not in the array

#### Intuition

* brute force, only 50 elements
* max is 50, max prefix sum 50^2
* check only the current position: n[i] == n[0] + i
* use bitmask instead of a set
* don't check sum if it is bigger than max

#### Approach

* kotlin: (sum...) - n.toSet() then take first
* rust: (n[0]..) gives the expected sequence

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun missingInteger(n: IntArray) =
    ((n.indices.takeWhile{n[it]==n[0]+it}.sumOf{n[it]}..2500)-n.toSet())[0]
```
```rust
    pub fn missing_integer(n: Vec<i32>) -> i32 {
        (n.iter().zip(n[0]..).take_while(|(a, b)| *a == b).map(|x| x.1).sum()..)
        .find(|x| !n.contains(x)).unwrap()
    }
```

