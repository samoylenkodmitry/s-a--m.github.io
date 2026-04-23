---
layout: leetcode-entry
title: "2402. Meeting Rooms III"
permalink: "/leetcode/problem/2025-12-27-2402-meeting-rooms-iii/"
leetcode_ui: true
entry_slug: "2025-12-27-2402-meeting-rooms-iii"
---

[2402. Meeting Rooms III](https://leetcode.com/problems/meeting-rooms-iii/description/) hard
[blog post](https://leetcode.com/problems/meeting-rooms-iii/solutions/7442963/kotlin-rust-by-samoylenkodmitry-zbi9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27122025-2402-meeting-rooms-iii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OKv9PxfkAXo)

![cc702986-a82b-4991-85ce-33ea0204d7ae (1).webp](/assets/leetcode_daily_images/e5f685e8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1217

#### Problem TLDR

Most frequent room 0..n for meetings [s,e] #hard #heap

#### Intuition

```j
    // 0 1 2 3 4 5 6 7 8 9 10
    // * * * * * * * * * * *   a
    //   * * * * *             b
    //           * * * * * *   b
    //                     * * a
    //
    // time is complicated
    //
    // some corner case i didn't see
    //
    // 0 1 2 3 4 5 6 7 8 9 10
    //   * * * * * * * * * *  a
    //     * * * * * * * * *  b
    //                     * * * * * * * *  a
    //                       * * * * * * *  b
    //                                   * * * * * * a
    //                                     * * * * * b
```

#### Approach

#### Complexity

- Time complexity:
$$O(nmlogm)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 269ms
    fun mostBooked(n: Int, m: Array<IntArray>): Int {
        m.sortBy { it[0] }; val f = IntArray(n); val t = LongArray(n)
        for ((s,e) in m) {
            val room = (0..<n).firstOrNull { t[it] <= s } ?: t.indexOf(t.min())
            t[room] = 1L*e + max(0, t[room]-s)
            ++f[room]
        }
        return f.indexOf(f.max())
    }
```
```rust
// 33ms
    pub fn most_booked(n: i32, mut m: Vec<Vec<i32>>) -> i32 {
        m.sort(); let (mut f, mut t, n) = ([0; 100], [0; 100], n as usize);
        for m in m {
            let (s, e, mut i, mut j, mut m) = (m[0] as i64, m[1] as i64, -1, -1, i64::MAX);
            for r in 0..n { if t[r] <= s { j = r as i32; break }; if t[r] < m { i = r as i32; m = t[r] }}
            let room = i.max(j) as usize; t[room] = e + 0.max(t[room] - s); f[room] += 1
        }
        (0..n).max_by_key(|&i| (f[i],Reverse(i))).unwrap() as _
    }
```

