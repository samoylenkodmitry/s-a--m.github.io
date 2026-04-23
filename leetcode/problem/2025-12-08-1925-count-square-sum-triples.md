---
layout: leetcode-entry
title: "1925. Count Square Sum Triples"
permalink: "/leetcode/problem/2025-12-08-1925-count-square-sum-triples/"
leetcode_ui: true
entry_slug: "2025-12-08-1925-count-square-sum-triples"
---

[1925. Count Square Sum Triples](https://leetcode.com/problems/count-square-sum-triples/description/) easy
[blog post](https://leetcode.com/problems/count-square-sum-triples/solutions/7399838/kotlin-rust-by-samoylenkodmitry-csii/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08122025-1925-count-square-sum-triples?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EFxlZn9oLvM)

![1a7fcc8f-43cd-4f69-94d4-012239f197d7 (1).webp](/assets/leetcode_daily_images/c33b91d8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1197

#### Problem TLDR

Count a^2+b^2=c^2 in 1..n range #easy

#### Intuition

O(n^3) is accepted
O(n^2): precompute n^2 numbers, lookup (a+b) in them

#### Approach

* we have to be in range 1..n, not in 1..250
* or check sqrt(a*a+b*b)

#### Complexity

- Time complexity:
$$O(n^3)$$ or O(n^2)

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 49ms
    fun countTriples(n: Int): Int {
        val s = (1..n).map { it * it }.toSet()
        return s.sumOf { a-> s.count { b-> (a+b) in s}}
    }
```
```rust
// 5ms
    pub fn count_triples(n: i32) -> i32 {
        (1..=n).map(|a| 2*(a..=n).filter(|b| {
        let c = (a*a+b*b).isqrt(); c <= n && c*c==a*a+b*b}).count() as i32).sum::<i32>()
    }
```

