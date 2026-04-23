---
layout: leetcode-entry
title: "3439. Reschedule Meetings for Maximum Free Time I"
permalink: "/leetcode/problem/2025-07-09-3439-reschedule-meetings-for-maximum-free-time-i/"
leetcode_ui: true
entry_slug: "2025-07-09-3439-reschedule-meetings-for-maximum-free-time-i"
---

[3439. Reschedule Meetings for Maximum Free Time I](https://leetcode.com/problems/reschedule-meetings-for-maximum-free-time-i/description) medium
[blog post](https://leetcode.com/problems/reschedule-meetings-for-maximum-free-time-i/solutions/6938033/kotlin-rust-by-samoylenkodmitry-brv2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/9072025-3439-reschedule-meetings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/miMjCJJEO70)
![1.webp](/assets/leetcode_daily_images/6401ea4a.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1044

#### Problem TLDR

Max free time after moving k events together #medium #sliding_window

#### Intuition

```j
    // .. ... .... ....
    // ..    ... ....       ....    ......
    //    a     b       c        d
    //   a+b    b+c       c+d                 k=1
    //    a+b+c  b+c+d                        k=2
    //    a+b+c+d                             k=3
```
Only the free intervals matter. Move `k+1` intervals together with sliding window.

#### Approach

* try to write O(1) memory solution
* corner cases are `start` and the `end`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(k)$$ or O(1)

#### Code

```kotlin

// 16ms
    fun maxFreeTime(evt: Int, k: Int, st: IntArray, et: IntArray): Int {
        val free = LinkedList<Int>(); var sum = 0
        return (0..st.size).maxOf { i ->
            val t = (if (i < st.size) st[i] else evt) - (if (i > 0) et[i - 1] else 0)
            sum += t; free += t; if (free.size > k + 1) sum -= free.removeFirst()
            sum
        }
    }

```
```kotlin

// 10ms
    fun maxFreeTime(evt: Int, k: Int, st: IntArray, et: IntArray): Int {
        var sum = 0
        return (0..st.size).maxOf { i ->
            sum += (if (i < st.size) st[i] else evt) - (if (i > 0) et[i - 1] else 0) -
            (if (i > k) st[i - k - 1] else 0) + if (i - k - 2 >= 0) et[i - k - 2] else 0
            sum
        }
    }

```
```rust

// 3ms
    pub fn max_free_time(evt: i32, k: i32, st: Vec<i32>, et: Vec<i32>) -> i32 {
        let (mut sum, k) = (0, k as usize);
        (0..=st.len()).map(|i| {
            sum += if i < st.len() { st[i] } else { evt } - if i > 0 { et[i - 1] } else { 0 } -
            if i > k { st[i - k - 1] } else { 0 } + if i >= k + 2 { et[i - k - 2] } else { 0 };
            sum
        }).max().unwrap_or(0)
    }

```
```c++

// 4ms
    int maxFreeTime(int evt, int k, vector<int>& st, vector<int>& et) {
        int sum = 0, res = 0, n = st.size();
        for (int i = 0; i <= n; ++i) res = max(res, sum +=
            (i < n ? st[i] : evt) -
            (i > 0 ? et[i - 1] : 0) -
            (i > k ? st[i - k - 1] : 0) +
            (i >= k + 2 ? et[i - k - 2] : 0));
        return res;
    }

```

