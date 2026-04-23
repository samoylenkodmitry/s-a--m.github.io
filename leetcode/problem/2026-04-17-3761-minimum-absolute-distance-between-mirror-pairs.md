---
layout: leetcode-entry
title: "3761. Minimum Absolute Distance Between Mirror Pairs"
permalink: "/leetcode/problem/2026-04-17-3761-minimum-absolute-distance-between-mirror-pairs/"
leetcode_ui: true
entry_slug: "2026-04-17-3761-minimum-absolute-distance-between-mirror-pairs"
---

[3761. Minimum Absolute Distance Between Mirror Pairs](https://leetcode.com/problems/minimum-absolute-distance-between-mirror-pairs/solutions/7955270/kotlin-rust-by-samoylenkodmitry-oanj/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17042026-3761-minimum-absolute-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7J81lVsCLGk)

![17.04.2026.webp](/assets/leetcode_daily_images/17.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1331

#### Problem TLDR

Distance to prev mirror number #medium

#### Intuition

Use a hashmap, scan, put mirrors into map, check for current number.

#### Approach

* Kotlin: string conversion is shorter
* Rust: (0..).zip(n), filter_map

#### Complexity

- Time complexity:
$$O(m)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 232ms
    fun minMirrorPairDistance(n: IntArray) = HashMap<Int, Int>().let { m ->
        n.indices.minOf { i ->
            i - (m[n[i]] ?: -n.size).also { m["${n[i]}".reversed().toInt()] = i }
        }.takeIf { it < n.size } ?: -1
    }
```
```rust
// 19ms
    pub fn min_mirror_pair_distance(n: Vec<i32>) -> i32 {
        let mut m = HashMap::new();
        (0..).zip(n).filter_map(|(i, mut x)| {
            let d = m.get(&x).map(|j| i-j);
            let mut r = 0; while x > 0 {r = r*10+x%10; x /= 10}
            m.insert(r, i); d
        }).min().unwrap_or(-1)
    }
```

