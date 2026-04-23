---
layout: leetcode-entry
title: "3433. Count Mentions Per User"
permalink: "/leetcode/problem/2025-12-12-3433-count-mentions-per-user/"
leetcode_ui: true
entry_slug: "2025-12-12-3433-count-mentions-per-user"
---

[3433. Count Mentions Per User](https://leetcode.com/problems/count-mentions-per-user/description) medium
[blog post](https://leetcode.com/problems/count-mentions-per-user/solutions/7408541/kotlin-rust-by-samoylenkodmitry-rc6d/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12122025-3433-count-mentions-per?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xctbYDDg_uA)

![2e36944c-7c92-45bc-8c37-d8b28e165d51 (1).webp](/assets/leetcode_daily_images/8fa2b674.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1201

#### Problem TLDR

Count messages and track online users #medium #simulation

#### Intuition

The problem is not that big. Sort by timestamp and put queries after offline modifications.

#### Approach

* use queue of coming online or just array of coming online times for all users

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 61ms
    fun countMentions(n: Int, e: List<List<String>>): IntArray {
        val r = IntArray(n); val on = IntArray(n)
        for ((type, ts, ids) in e.sortedBy { it[1].toInt()*100-it[0][0].toInt() })
            if (type[0] == 'O') on[ids.toInt()] = ts.toInt() + 60
            else if (ids[0] == 'A') for (i in 0..<n) ++r[i]
            else if (ids[0] == 'i') for (i in ids.split(" ")) ++r[i.drop(2).toInt()]
            else for (i in 0..<n) if (on[i]<=ts.toInt()) ++r[i]
        return r
    }
```
```rust
// 0ms
    pub fn count_mentions(n: i32, mut e: Vec<Vec<String>>) -> Vec<i32> {
        let n = n as usize; let (mut r, mut o) = (vec![0; n], vec![0; n]);
        e.sort_unstable_by_key(|v| v[1].parse::<i32>().unwrap()*100-v[0].as_bytes()[0] as i32);
        for v in &e {
            let (tp, ts) = (v[0].as_bytes()[0], v[1].parse::<i32>().unwrap());
            if tp == b'O' { o[v[2].parse::<usize>().unwrap()] = ts + 60 }
            else if v[2].as_bytes()[0] == b'A' { for i in &mut r { *i += 1 }}
            else if v[2].as_bytes()[0] == b'i' { for i in v[2].split(" ") { r[i[2..].parse::<usize>().unwrap()] += 1 }}
            else { for i in 0..n { if o[i] <= ts { r[i] += 1 }}}
        } r
    }
```

