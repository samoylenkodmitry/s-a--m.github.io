---
layout: leetcode-entry
title: "1189. Maximum Number of Balloons"
permalink: "/leetcode/problem/2026-06-22-1189-maximum-number-of-balloons/"
leetcode_ui: true
entry_slug: "2026-06-22-1189-maximum-number-of-balloons"
---

[1189. Maximum Number of Balloons](https://leetcode.com/problems/maximum-number-of-balloons/solutions/8351008/kotlin-rust-by-samoylenkodmitry-17ug/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22062026-1189-maximum-number-of-balloons?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6UzICTD3ATU)

https://dmitrysamoylenko.com/leetcode/

![22.06.2026.webp](/assets/leetcode_daily_images/22.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1398

#### Problem TLDR

Count "balloon"s in stirng

#### Intuition

Calculate the frequency; Divide the frequency of 'l' and 'o' by 2.

#### Approach

* we can iterate 5 times

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxNumberOfBalloons(t: String) =
    (0..4).minOf {t.count{c->c=="balon"[it]}/(it/2%2+1)}
```
```rust
    pub fn max_number_of_balloons(t: String) -> i32 {
        "balon".chars().map(|c|
        t.matches(c).count()/((c=='l'||c=='o')as usize+1)).min().unwrap() as _
    }
```

