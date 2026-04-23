---
layout: leetcode-entry
title: "3440. Reschedule Meetings for Maximum Free Time II"
permalink: "/leetcode/problem/2025-07-10-3440-reschedule-meetings-for-maximum-free-time-ii/"
leetcode_ui: true
entry_slug: "2025-07-10-3440-reschedule-meetings-for-maximum-free-time-ii"
---

[3440. Reschedule Meetings for Maximum Free Time II](https://leetcode.com/problems/reschedule-meetings-for-maximum-free-time-ii/description/) medium
[blog post](https://leetcode.com/problems/reschedule-meetings-for-maximum-free-time-ii/solutions/6941810/kotlin-rust-by-samoylenkodmitry-bpfl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10072025-3440-reschedule-meetings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hY2uEOdgCIM)
![1.webp](/assets/leetcode_daily_images/4ac7188f.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1045

#### Problem TLDR

Max free time after moving 1 event #medium #sorting

#### Intuition

```j

    // 0  17..19  24..25   41
    //  17      5        16
    //      2       1
```

    * look for all free windows
    * look around each event
    * look if each event can fit into another window
    * sort free windows
    * track windows indices

Space optimization: do forward and backward pass to track the largest seen gap.

#### Approach

* we can do forward & backward pass in a single loop

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

// 91ms
    fun maxFreeTime(evt: Int, st: IntArray, et: IntArray): Int {
        val free = IntArray(st.size)
        val gaps = Array(3) { 0 to 0 }; gaps[0] = st[0] to 0
        for (i in 0..<st.size) {
            val s = st[i]; val e = et[i]
            val prev = if (i == 0) 0 else et[i - 1]
            val before = s - prev
            val next = if (i < st.size - 1) st[i + 1] else evt
            val after = next - e
            free[i] = before + after
            val g = (0..2).minBy { gaps[it].first }
            if (gaps[g].first < after) gaps[g] = after to (i + 1)
        }
        var res = 0
        for (i in free.indices) {
            var fit = 0
            val curr = et[i] - st[i]
            for (g in 0..2) if (gaps[g].first >= curr && gaps[g].second !in i..i+1)
                fit = curr
            res = max(res, free[i] + fit)
        }
        return res
    }

```
```kotlin

// 7ms
    fun maxFreeTime(evt: Int, st: IntArray, et: IntArray): Int {
        var left = 0; var right = 0; var res = 0; var j = st.size - 1
        for (i in st.indices) {
            var before = st[i] - if (i == 0) 0 else et[i - 1]
            var after = (if (i < st.size - 1) st[i + 1] else evt) - et[i]
            res = max(res, before + after + if (et[i] - st[i] <= left) et[i] - st[i] else 0)
            left = max(left, before)
            before = (if (j == st.size - 1) evt else st[j + 1]) - et[j]
            after = st[j] - if (j > 0) et[j - 1] else 0
            res = max(res, before + after + if (et[j] - st[j] <= right) et[j] - st[j] else 0)
            right = max(right, before); j--
        }
        return res
    }

```
```rust

// 4ms
    pub fn max_free_time(evt: i32, st: Vec<i32>, et: Vec<i32>) -> i32 {
        let (mut left, mut right, mut res, mut j) = (0, 0, 0, st.len() - 1);
        for i in 0..st.len() {
            let before = st[i] - if i == 0 { 0 } else { et[i - 1] };
            let after = (if i < st.len() - 1 { st[i + 1] } else { evt }) - et[i];
            res = res.max(before + after + if et[i] - st[i] <= left { et[i] - st[i] } else { 0 });
            left = left.max(before);
            let before = (if j == st.len() - 1 { evt } else { st[j + 1] }) - et[j];
            let after = st[j] - if j > 0 { et[j - 1] } else { 0 };
            res = res.max(before + after + if et[j] - st[j] <= right { et[j] - st[j] } else { 0 });
            right = right.max(before); j -= 1
        } res
    }

```
```c++

// 0ms
    int maxFreeTime(int evt, vector<int>& st, vector<int>& et) {
        int left = 0, right = 0, res = 0, j = st.size() - 1;
        for (int i = 0; i < st.size(); ++i) {
            int before = st[i] - (i == 0 ? 0 : et[i - 1]);
            int after = (i < st.size() - 1 ? st[i + 1] : evt) - et[i];
            int dur = et[i] - st[i];
            res = max(res, before + after + (dur <= left ? dur : 0));
            left = max(left, before);
            before = (j == st.size() - 1 ? evt : st[j + 1]) - et[j];
            after = st[j] - (j > 0 ? et[j - 1] : 0);
            dur = et[j] - st[j];
            res = max(res, before + after + (dur <= right ? dur : 0));
            right = max(right, before);
            j--;
        }
        return res;
    }

```

