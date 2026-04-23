---
layout: leetcode-entry
title: "852. Peak Index in a Mountain Array"
permalink: "/leetcode/problem/2023-07-25-852-peak-index-in-a-mountain-array/"
leetcode_ui: true
entry_slug: "2023-07-25-852-peak-index-in-a-mountain-array"
---

[852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/description/) medium
[blog post](https://leetcode.com/problems/peak-index-in-a-mountain-array/solutions/3812258/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/25072023-852-peak-index-in-a-mountain?sd=pf)
![image.png](/assets/leetcode_daily_images/c77699f0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/286

#### Problem TLDR

Mountain pattern `index` in the array in log time

#### Intuition

Do the Binary Search of the biggest growing index

#### Approach

For more robust Binary Search code:
* use inclusive `lo` and `hi`
* do the last check `lo == hi`
* always write the result `ind = mid` if conditions are met
* always move the borders `lo = mid - 1`, `hi = mid + 1`

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun peakIndexInMountainArray(arr: IntArray): Int {
        var lo = 1
        var hi = arr.lastIndex
        var ind = -1
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          if (arr[mid] > arr[mid - 1]) {
            ind = mid
            lo = mid + 1
          } else hi = mid - 1
        }
        return ind
    }

```
#### Magical Rundown

```
🌄 "Look at that crimson blush, Alpha!" A radiant sunrise anoints the
towering Everest, its snow-capped peaks aglow with the day's first
light. An ethereal landscape, a symphony of shadows and silhouettes,
lays the stage for an impending adventure. 🏔️

Team Alpha 🥾 chuckles, their voices swallowed by the wind, "Today's
the day we've been dreaming of, Charlie!" Team Charlie 🦅, encased
in their mountain gear, share their excitement. Their eyes, reflecting
the sunlit peaks, are fixated on the summit – their celestial goal.

Base Camps (BC):
0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Everest Heights:
1K  2K  3K  4K  5K  6K  7K  8K  9K 10K 11K 12K 13K 14K 15K 16K 15K 14K 13K 12K 11K
    🥾(Team Alpha)                  🏔️(Mysterious Mid Point)                    🦅(Team Charlie)

🧭 "We're off to conquer the Everest!" Alpha's voice reverberates
with a hopeful intensity. Their strategy, an intricate dance with
numbers and ambition – Binary Search. The mountain, its snow-capped
peaks reaching for the skies, hums ancient tales to their eager ears.

BC:        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Heights:  1K  2K  3K  4K  5K  6K  7K  8K  9K 10K 11K 12K 13K 14K 15K 16K 15K 14K 13K 12K 11K
                                                 🥾🏁(Team Alpha's Milestone)            🦅(Team Charlie)

"10K! Feels like we've captured a bit of heaven," Team Alpha shares
their awe, their voices a mere whisper against the grandeur of the
landscape.

🏞️ But the mountain, a grand enigma, hides her secrets well...

With a sudden, heart-stopping rumble, the mountain shivers, the
seemingly innocuous snow beneath Team Charlie's feet giving way. A
fierce avalanche sweeps down the slopes, sending Charlie scrambling
for cover.

BC:        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Heights:  1K  2K  3K  4K  5K  6K  7K  8K  9K 10K 11K 12K 13K 14K 15K 16K 15K 14K 13K 12K 11K
                                                 🥾🏁(Team Alpha's Resolve)              ❄️🦅(Team Charlie's Setback)

"Avalanche!" Charlie's voice, choked with frosty fear, crackles over
the radio. Yet, Team Alpha, undeterred by the wrath of the mountain,
pushes forward. "Hold tight, Charlie! We're stardust-bound!" They
continue their daring ascent, breaching the cloudline to a dizzying
16K.

🚩 "Charlie, we're among the stars!" Alpha's voice, filled with
joyous triumph, echoes through the radio. The peak is conquered.

BC:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Heights:  1K  2K  3K  4K  5K  6K  7K  8K  9K 10K 11K 12K 13K 14K 15K 16K 15K 14K 13K 12K 11K
                                                                🚩🎉(Victorious Summit at BC 15!)

As the sun bathes the snowy peaks in a golden hue, Team Alpha plants
their triumphant flag at the top. They stand there, at the roof of
the world, their hearts swelling with joy and pride. It's the journey,
the shared aspirations, the dream of reaching for the stars, that truly
defines their adventure. 🌠
```

