---
layout: leetcode-entry
title: "3731. Find Missing Elements"
permalink: "/leetcode/problem/2026-08-04-3731-find-missing-elements/"
leetcode_ui: true
entry_slug: "2026-08-04-3731-find-missing-elements"
---

[3731. Find Missing Elements](https://leetcode.com/problems/find-missing-elements/solutions/8440108/kotlin-rust-by-samoylenkodmitry-sf7r/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04082026-3731-find-missing-elements?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/10XdpCEky20)

https://dmitrysamoylenko.com/leetcode/

![04.08.2026.webp](/assets/leetcode_daily_images/04.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1441

#### Problem TLDR

Find missing numbers in range min..max

#### Intuition

Brute-force. Or convert to set to improve time complexity. But its only 100 elements.

#### Approach

* Kotlin: just subtract set from range
* Rust: use itertools minmax

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun findMissingElements(n: IntArray)=
    (n.min()..n.max())-n.toSet()
```
```rust
    pub fn find_missing_elements(n: Vec<i32>) -> Vec<i32> {
        let MinMax(a,b) = n.iter().minmax() else { panic!() };
        (*a..*b).filter(|x|!n.contains(x)).collect()
    }
```

