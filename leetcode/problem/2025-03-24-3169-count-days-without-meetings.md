---
layout: leetcode-entry
title: "3169. Count Days Without Meetings"
permalink: "/leetcode/problem/2025-03-24-3169-count-days-without-meetings/"
leetcode_ui: true
entry_slug: "2025-03-24-3169-count-days-without-meetings"
---

[3169. Count Days Without Meetings](https://leetcode.com/problems/count-days-without-meetings/description) medium
[blog post](https://leetcode.com/problems/count-days-without-meetings/solutions/6572822/kotlin-rust-by-samoylenkodmitry-z8fg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24032025-3169-count-days-without?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/c0U-49VvWlE)
![1.webp](/assets/leetcode_daily_images/6f8a6f1f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/937

#### Problem TLDR

Days of non-intersecting intervals #medium #line_sweep

#### Intuition

Several ways:
* line sweep with heap
* line sweep with TreeMap
* sorting intervals

#### Approach

* careful with of-by-one in the start and in the end

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$ or O(n)

#### Code

```kotlin

    fun countDays(days: Int, m: Array<IntArray>): Int {
        m.sortBy { it[0] }; var d = 0; var res = 0
        for ((s, e) in m) { res += max(0, s - d - 1); d = max(d, e) }
        return res + days - d
    }

```
```kotlin

    fun countDays(days: Int, m: Array<IntArray>): Int {
        val w = TreeMap<Int, Int>(); var d = 0; var cnt = 0; var res = 0
        for ((s, e) in m) { w[s] = 1 + (w[s] ?: 0); w[e + 1] = -1 + (w[e + 1] ?: 0) }
        for ((s, v) in w) { if (cnt == 0) res += s - d; cnt += v; d = s }
        return res + days - d
    }

```
```kotlin

    fun countDays(days: Int, meetings: Array<IntArray>): Int {
        val q = PriorityQueue<List<Int>>(compareBy({ it[0] }, { -it[1] }))
        for ((s, e) in meetings) { q += listOf(s, 1); q += listOf(e, -1) }
        var d = 0; var cnt = 0; var res = 0
        while (q.size > 0) {
            val (day, delta) = q.poll()
            if (cnt == 0) res += day - d - 1
            cnt += delta; d = day
        }
        return res + days - d
    }

```
```rust

    pub fn count_days(days: i32, mut m: Vec<Vec<i32>>) -> i32 {
        m.sort_unstable(); let (mut d, mut r) = (0, 0);
        for x in m { r += 0.max(x[0] - d - 1); d = d.max(x[1]) }
        r + days - d
    }

```
```c++

int countDays(int days, vector<vector<int>> m) {
    sort(m.begin(), m.end()); int d = 0, r = 0;
    for (auto& x : m) r += max(0, x[0] - d - 1), d = max(d, x[1]);
    return r + days - d;
}

```

