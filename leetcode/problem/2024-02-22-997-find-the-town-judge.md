---
layout: leetcode-entry
title: "997. Find the Town Judge"
permalink: "/leetcode/problem/2024-02-22-997-find-the-town-judge/"
leetcode_ui: true
entry_slug: "2024-02-22-997-find-the-town-judge"
---

[997. Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/description/) easy
[blog post](https://leetcode.com/problems/find-the-town-judge/solutions/4765796/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22022024-997-find-the-town-judge?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/X0ei_8ygmUk)
![image.png](/assets/leetcode_daily_images/63ab84d5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/515

#### Problem TLDR

Find who trusts nobody and everybody trusts him in [trust, trusted] array.

#### Intuition

First, potential judge is from set `1..n` excluding all the people who trust someone `trust.map { it[0] }`.
Next, check everybody trust him `count == n - 1`.

Another approach, is to count in-degree and out-degree nodes in graph.

#### Approach

For the second approach, we didn't need to count out-degrees, just make in-degrees non-usable.

Let's try to shorten the code.
* Kotlin: use `toSet`, `map`, `takeIf`, `count`, `first`
* Rust: `find`, `map_or`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findJudge(n: Int, trust: Array<IntArray>) =
    ((1..n).toSet() - trust.map { it[0] }.toSet())
      .takeIf { it.size == 1 }?.first()
      ?.takeIf { j ->
        trust.count { it[1] == j } == n - 1
      } ?: -1

```
```rust

  pub fn find_judge(n: i32, trust: Vec<Vec<i32>>) -> i32 {
    let mut deg = vec![0; n as usize + 1];
    for e in trust {
      deg[e[0] as usize] += n;
      deg[e[1] as usize] += 1;
    }
    (1..deg.len()).find(|&j| deg[j] == n - 1).map_or(-1, |j| j as i32)
  }

```

