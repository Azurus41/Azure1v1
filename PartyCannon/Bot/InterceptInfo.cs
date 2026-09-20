using System;
using RedUtils;
using RedUtils.Math;

namespace Bot;

public class InterceptInfo
{
    public float OurBestTime { get; set; }

    public Car OurBestCar { get; set; }

    public Vec3 OurBestBallPosition { get; set; }

    public float OpponentBestTime { get; set; }

    public Car OpponentBestCar { get; set; }

    public Vec3 OpponentBestBallPosition { get; set; }

    public bool OpponentsAreFaster { get; set; }

    public bool WeAreFaster { get; set; }

    public float TimeDifference => OpponentBestTime - OurBestTime;

    public float TimeAdvantage => Math.Max(0f, TimeDifference);
}
